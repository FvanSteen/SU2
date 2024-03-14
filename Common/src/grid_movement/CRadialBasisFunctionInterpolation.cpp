/*!
 * \file CRadialBasisFunctionInterpolation.cpp
 * \brief Subroutines for moving mesh volume elements using Radial Basis Function interpolation.
 * \author F. van Steen
 * \version 8.0.1 "Harrier"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2023, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "../../include/grid_movement/CRadialBasisFunctionInterpolation.hpp"
#include "../../include/interface_interpolation/CRadialBasisFunction.hpp"
#include "../../include/toolboxes/geometry_toolbox.hpp"
#include "../../include/adt/CADTPointsOnlyClass.hpp"


CRadialBasisFunctionInterpolation::CRadialBasisFunctionInterpolation(CGeometry* geometry, CConfig* config) : CVolumetricMovement(geometry) {
  /*--- Retrieve type of RBF and if applicable its support radius ---*/
  kindRBF =  config->GetKindRadialBasisFunction();
  radius = config->GetRadialBasisFunctionParameter();
  
  DataReduction = config->GetRBF_DataReduction();
  
  if(DataReduction){
    controlNodes = &greedyNodes;
    GreedyTolerance = config->GetRBF_GreedyTolerance();
    GreedyCorrectionFactor = config->GetRBF_GreedyCorrectionFactor();
  }else{
    controlNodes = &boundaryNodes;
  }

}

CRadialBasisFunctionInterpolation::~CRadialBasisFunctionInterpolation(void) = default;

void CRadialBasisFunctionInterpolation::SetVolume_Deformation(CGeometry* geometry, CConfig* config, bool UpdateGeo, bool Derivative,
                                                bool ForwardProjectionDerivative){
  su2double MinVolume, MaxVolume;

  /*--- Retrieving number of deformation steps and screen output from config ---*/

  auto Nonlinear_Iter = config->GetGridDef_Nonlinear_Iter();

  auto Screen_Output = config->GetDeform_Output();
  
  /*--- Disable the screen output if we're running SU2_CFD ---*/

  if (config->GetKind_SU2() == SU2_COMPONENT::SU2_CFD && !Derivative) Screen_Output = false;
  if (config->GetSmoothGradient()) Screen_Output = true;

  /*--- Assigning the node types ---*/
  SetControlNodes(geometry, config);
  SetInternalNodes(geometry);

  /*--- Looping over the number of deformation iterations ---*/
  for (auto iNonlinear_Iter = 0ul; iNonlinear_Iter < Nonlinear_Iter; iNonlinear_Iter++) {
    cout << "deformation step: " << iNonlinear_Iter << endl;
    /*--- Compute min volume in the entire mesh. ---*/

    ComputeDeforming_Element_Volume(geometry, MinVolume, MaxVolume, Screen_Output);
    if (rank == MASTER_NODE && Screen_Output)
      cout << "Min. volume: " << MinVolume << ", max. volume: " << MaxVolume << "." << endl;

    /*--- Obtaining the interpolation coefficients of the control nodes ---*/
    GetInterpolationCoefficients(geometry, config, iNonlinear_Iter);
    
    /*--- Updating the coordinates of the grid ---*/
    UpdateGridCoord(geometry, config);

    if(UpdateGeo){
      UpdateDualGrid(geometry, config);
    }

    /*--- Check for failed deformation (negative volumes). ---*/

    ComputeDeforming_Element_Volume(geometry, MinVolume, MaxVolume, Screen_Output);

    /*--- Calculate amount of nonconvex elements ---*/

    ComputenNonconvexElements(geometry, Screen_Output);

    if (rank == MASTER_NODE && Screen_Output) {
      cout << "Non-linear iter.: " << iNonlinear_Iter + 1 << "/" << Nonlinear_Iter << ". ";
      if (nDim == 2)
        cout << "Min. area: " << MinVolume <<  "." << endl;
      else
        cout << "Min. volume: " << MinVolume <<  "." << endl;
    }
  }
}

void CRadialBasisFunctionInterpolation::GetInterpolationCoefficients(CGeometry* geometry, CConfig* config, unsigned long iNonlinear_Iter){

  if(DataReduction){

    GreedyIteration(geometry, config);
  }else{
    /*--- Deformation vector only has to be set once ---*/
    if(iNonlinear_Iter == 0){
      SetDeformationVector(geometry, config);
    }

    /*--- Computing the interpolation matrix with RBF evaluations based on Euclidean distance ---*/
    SetInterpolationMatrix(geometry, config);

    /*--- Solving the RBF system to get the interpolation coefficients ---*/
    SolveRBF_System();   
  }
}


void CRadialBasisFunctionInterpolation::SetControlNodes(CGeometry* geometry, CConfig* config){
  unsigned short iMarker; 
  unsigned short iVertex; 
  
   if(DataReduction){
    greedyNodes.resize(0);
    nGreedyNodes = 0;
  }

  /*--- Total number of boundary nodes (including duplicates of shared boundaries) ---*/
  unsigned long nBoundNodes = 0;
  for(iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++){
      nBoundNodes += geometry->nVertex[iMarker];
  }

  /*--- Vector with boudary nodes has at most nBoundNodes ---*/
  boundaryNodes.resize(nBoundNodes);

  /*--- Storing of the global, marker and vertex indices ---*/
  unsigned long count = 0;
  for(iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++){
    for(iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++){
      boundaryNodes[count+iVertex] = new CRadialBasisFunctionNode(geometry, iMarker, iVertex);
      }
    count += geometry->nVertex[iMarker];
  }

  /*--- Sorting of the boundary nodes based on global index ---*/
  sort(boundaryNodes.begin(), boundaryNodes.end(), Compare);

  /*--- Obtaining unique set of boundary nodes ---*/
  boundaryNodes.resize(std::distance(boundaryNodes.begin(), unique(boundaryNodes.begin(), boundaryNodes.end(), Equal)));
  
  /*--- Updating the number of boundary nodes ---*/
  nBoundaryNodes = boundaryNodes.size();
  
}

void CRadialBasisFunctionInterpolation::SetInterpolationMatrix(CGeometry* geometry, CConfig* config){
  unsigned long iNode, jNode;

  /*--- Initialization of the interpolation matrix ---*/
  interpMat.Initialize(controlNodes->size());

  /*--- Construction of the interpolation matrix. 
    Since this matrix is symmetric only upper halve has to be considered ---*/

  /*--- Looping over the target nodes ---*/
  for(iNode = 0; iNode < controlNodes->size(); iNode++ ){
    /*--- Looping over the control nodes ---*/
    for (jNode = iNode; jNode < controlNodes->size(); jNode++){

      /*--- Distance between nodes ---*/
      auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes)[iNode]->GetIndex()), geometry->nodes->GetCoord((*controlNodes)[jNode]->GetIndex()));
      
      /*--- Evaluation of RBF ---*/
      interpMat(iNode, jNode) = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));
    }
  }

  /*--- Obtaining lower halve using symmetry ---*/
  const bool kernelIsSPD = (kindRBF == RADIAL_BASIS::WENDLAND_C2) || (kindRBF == RADIAL_BASIS::GAUSSIAN) ||
                          (kindRBF == RADIAL_BASIS::INV_MULTI_QUADRIC);
  interpMat.Invert(kernelIsSPD);

}

void CRadialBasisFunctionInterpolation::SetDeformationVector(CGeometry* geometry, CConfig* config){
  /* --- Initialization of the deformation vector ---*/
  deformationVector.resize(controlNodes->size()*nDim, 0.0);

  /*--- If requested (no by default) impose the surface deflections in
    increments and solve the grid deformation with
    successive small deformations. ---*/
  su2double VarIncrement = 1.0 / ((su2double)config->GetGridDef_Nonlinear_Iter());

  /*--- Setting nonzero displacements of the moving markers ---*/
  for(auto i = 0; i < controlNodes->size(); i++){
    if(config->GetMarker_All_Moving((*controlNodes)[i]->GetMarker())){
      for(auto iDim = 0; iDim < nDim; iDim++){
        deformationVector[i+iDim*controlNodes->size()] = SU2_TYPE::GetValue(geometry->vertex[(*controlNodes)[i]->GetMarker()][(*controlNodes)[i]->GetVertex()]->GetVarCoord()[iDim] * VarIncrement);
        }
      }else{
        for(auto iDim = 0; iDim < nDim; iDim++){
        deformationVector[i+iDim*controlNodes->size()] = 0.0;
        }
      }
  }  
}

void CRadialBasisFunctionInterpolation::SetInternalNodes(CGeometry* geometry){

  /*--- resizing the internal nodes vector ---*/
  nInternalNodes = geometry->GetnPoint() - nBoundaryNodes;
  internalNodes.resize(nInternalNodes);

  /*--- Looping over all nodes and check if present in boundary nodes vector ---*/
  unsigned long idx_cnt = 0, idx_control = 0;
  for(unsigned long iNode = 0; iNode < geometry->GetnPoint(); iNode++){
    
    /*--- If iNode is equal to boundaryNodes[idx_control] 
      then this node is a boundary node and idx_control can be updated ---*/
    if(idx_control < nBoundaryNodes && iNode == boundaryNodes[idx_control]->GetIndex()){idx_control++;}
    
    /*--- If not equal then the node is an internal node ---*/
    else{
      internalNodes[idx_cnt] = iNode;
      idx_cnt++;
    }   
  }  
}


void CRadialBasisFunctionInterpolation::SolveRBF_System(){

  /*--- resizing the interpolation coefficient vector ---*/
  coefficients.resize(nDim*controlNodes->size());

  /*--- Looping through the dimensions in order to find the interpolation coefficients for each direction ---*/
  unsigned short iDim;
  for(iDim = 0; iDim < nDim; iDim++){
    interpMat.MatVecMult(deformationVector.begin()+iDim*controlNodes->size(), coefficients.begin()+iDim*controlNodes->size());
  }

  cout << endl;
}

void CRadialBasisFunctionInterpolation::UpdateGridCoord(CGeometry* geometry, CConfig* config){
  cout << "updating the grid coordinates" << endl;
  unsigned long iNode, cNode;
  unsigned short iDim;
  
  /*--- Vector for storing the coordinate variation ---*/
  su2double var_coord[nDim]{0.0};
  
  /*--- Loop over the internal nodes ---*/
  for(iNode = 0; iNode < nInternalNodes; iNode++){

    /*--- Loop for contribution of each control node ---*/
    for(cNode = 0; cNode < controlNodes->size(); cNode++){
      
      /*--- Determine distance between considered internal and control node ---*/
      auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes)[cNode]->GetIndex()), geometry->nodes->GetCoord(internalNodes[iNode]));

      /*--- Evaluate RBF based on distance ---*/
      auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));

      /*--- Add contribution to total coordinate variation ---*/
      for( iDim = 0; iDim < nDim; iDim++){
        var_coord[iDim] += rbf*coefficients[cNode + iDim*controlNodes->size()];
      }
    }

    /*--- Apply the coordinate variation and resetting the var_coord vector to zero ---*/
    for(iDim = 0; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(internalNodes[iNode], iDim, var_coord[iDim]);
      var_coord[iDim] = 0;
    } 
  }  

  if(DataReduction){
    // setting the coords of the boundary nodes (non-control) in case of greedy
    for(iNode = 0; iNode < nBoundaryNodes; iNode++){
      for(cNode = 0; cNode < controlNodes->size(); cNode++){
    
      auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes)[cNode]->GetIndex()), geometry->nodes->GetCoord(boundaryNodes[iNode]->GetIndex()));

      auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));
  
      for( iDim = 0; iDim < nDim; iDim++){
        var_coord[iDim] += rbf*coefficients[cNode + iDim*controlNodes->size()];
      }
    }
  
    for(iDim = 0; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(boundaryNodes[iNode]->GetIndex(), iDim, var_coord[iDim]);
      var_coord[iDim] = 0;
    }

      // auto err = boundaryNodes[iNode]->GetError();
      // // cout << boundaryNodes[iNode]->GetIndex() << '\t' << err[0] << '\t' << err[1] << endl;
      // for(iDim = 0; iDim < nDim; iDim++){
      //     geometry->nodes->AddCoord(boundaryNodes[iNode]->GetIndex(), iDim, -err[iDim]);
      // }

    }
  }
  
  /*--- Applying the surface deformation, which are stored in the deformation vector ---*/
  for(cNode = 0; cNode < controlNodes->size(); cNode++){
    if(config->GetMarker_All_Moving((*controlNodes)[cNode]->GetMarker())){
      for(iDim = 0; iDim < nDim; iDim++){
        geometry->nodes->AddCoord((*controlNodes)[cNode]->GetIndex(), iDim, deformationVector[cNode + iDim*controlNodes->size()]);
      }
    }
  }

  if(DataReduction){
    SetCorrection(geometry);
  }

}

void CRadialBasisFunctionInterpolation::GreedyIteration(CGeometry* geometry, CConfig* config){

  GetInitMaxErrorNode(geometry);
  MaxError = GeometryToolbox::Norm(nDim, geometry->vertex[boundaryNodes[MaxErrorNode]->GetMarker()][boundaryNodes[MaxErrorNode]->GetVertex()]->GetVarCoord());
  cout << "FOUND MAX ERROR: " << MaxError << endl;
  MaxError = 1;
  unsigned short greedy_iter = 0;
  while(MaxError > GreedyTolerance){
    greedy_iter++;
    cout << "iteration: " << greedy_iter << endl;
    AddControlNode(geometry);

    SetDeformationVector(geometry, config);

    SetInterpolationMatrix(geometry, config);

    SolveRBF_System();

    MaxError = GetError(geometry, config);
    // cout << MaxError << '\t' << boundaryNodes[MaxErrorNode]->GetIndex() << '\t' << boundaryNodes[MaxErrorNode]->GetError()[0] << '\t' << boundaryNodes[MaxErrorNode]->GetError()[1] << endl;  
  }
}

void CRadialBasisFunctionInterpolation::GetInitMaxErrorNode(CGeometry* geometry){
  unsigned short iNode;

  su2double maxDeformation = 0.0;
  su2double normDeformation;

  for(iNode = 0; iNode < nBoundaryNodes; iNode++){
    normDeformation = GeometryToolbox::Norm(nDim, geometry->vertex[boundaryNodes[iNode]->GetMarker()][boundaryNodes[iNode]->GetVertex()]->GetVarCoord());
    if(normDeformation > maxDeformation){
      maxDeformation = normDeformation;
      MaxErrorNode = iNode;
    }      
  }
}

void CRadialBasisFunctionInterpolation::AddControlNode(CGeometry* geometry){
  // cout << "Added node: " <<  boundaryNodes[MaxErrorNode]->GetIndex() << endl;
  // addition of control node
  greedyNodes.push_back(move(boundaryNodes[MaxErrorNode]));
  nGreedyNodes++;

  // removing node from the set of boundary nodes
  boundaryNodes.erase(boundaryNodes.begin()+MaxErrorNode);
  nBoundaryNodes--;

}

su2double CRadialBasisFunctionInterpolation::GetError(CGeometry* geometry, CConfig* config){
  unsigned long iNode;
  unsigned short iDim;
  su2double localError[nDim];

  su2double error = 0.0, errorMagnitude;

  for(iNode = 0; iNode < nBoundaryNodes; iNode++){

    boundaryNodes[iNode]->SetError(GetNodalError(geometry, config, iNode, localError), nDim);

    errorMagnitude = GeometryToolbox::Norm(nDim, localError);
    if(errorMagnitude > error){
      error = errorMagnitude;
      MaxErrorNode = iNode;
    }
  }  
  return error;
}

su2double* CRadialBasisFunctionInterpolation::GetNodalError(CGeometry* geometry, CConfig* config, unsigned long iNode, su2double* localError){ 
  unsigned short iDim;
  su2double* displacement;
  if(config->GetMarker_All_Moving(boundaryNodes[iNode]->GetMarker())){
    displacement = geometry->vertex[boundaryNodes[iNode]->GetMarker()][boundaryNodes[iNode]->GetVertex()]->GetVarCoord();
    // cout << boundaryNodes[iNode]->GetIndex() << '\t' << displacement[0] << '\t' << displacement[1] << endl;  
    for(iDim = 0; iDim < nDim; iDim++){
      localError[iDim] = -displacement[iDim];
    }
  }else{
    for(iDim = 0; iDim < nDim; iDim++){
      localError[iDim] = 0;
    }
  }

  for(auto cNode = 0; cNode < controlNodes->size(); cNode++){
    auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes)[cNode]->GetIndex()), geometry->nodes->GetCoord(boundaryNodes[iNode]->GetIndex()));

    auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));

    for( iDim = 0; iDim < nDim; iDim++){
      localError[iDim] += rbf*coefficients[cNode + iDim*controlNodes->size()];
    }
  }
  // cout << boundaryNodes[iNode]->GetIndex() << '\t' << localError[0] << '\t' << localError[1] << endl;
  return localError;
}

void CRadialBasisFunctionInterpolation::SetCorrection(CGeometry* geometry){
  unsigned long iVertex, iNode, iDim, i, j, pointID;
  unsigned long nVertexBound = nBoundaryNodes;
  su2double dist;
  vector<su2double> Coord_bound(nDim*nVertexBound);
  vector<unsigned long> PointIDs(nVertexBound);
  int rankID;

  su2double CorrectionRadius = GreedyCorrectionFactor*MaxError;

  i = 0;
  j = 0;
  for(iVertex = 0; iVertex < nVertexBound; iVertex++){
    iNode = boundaryNodes[iVertex]->GetIndex();
    PointIDs[i++] = iVertex;
    for(iDim = 0; iDim < nDim; iDim++){
      Coord_bound[j++] = geometry->nodes->GetCoord(iNode, iDim);
    }
  }

  CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), true);

  for(iNode = 0; iNode < nInternalNodes; iNode++){
    BoundADT.DetermineNearestNode(geometry->nodes->GetCoord(internalNodes[iNode]), dist, pointID, rankID);  
    auto err = boundaryNodes[pointID]->GetError();
    // cout << internalNodes[iNode] << '\t' << dist << '\t'  << boundaryNodes[pointID]->GetIndex() << '\t' << err[0] <<'\t' << err[1] << endl;
    auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, CorrectionRadius, dist));
    for(iDim = 0; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(internalNodes[iNode], iDim, rbf*err[iDim]);
    }
  }

  for(iNode = 0; iNode < nBoundaryNodes; iNode++){
    auto err =  boundaryNodes[iNode]->GetError();
    for(iDim = 0; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(boundaryNodes[iNode]->GetIndex(), iDim, -err[iDim]);
    }
  }
}