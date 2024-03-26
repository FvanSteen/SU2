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
  
  controlNodes = &boundaryNodes;
  if(1){//TODO -> if BL preservation
    controlNodes = &wallNodes;
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
  // NOTE Adding the BL nodes at the end of the controlNodes vector does not work since then it will go wrong when determining the internal nodes
  // therefore to get the accurate internal nodes all nodes should be included in the controlNodes and somehow the wall and bl nodes should be seperated.
  SetControlNodes(geometry, config);
  SetInternalNodes(geometry);
  
  /*--- Looping over the number of deformation iterations ---*/
  for (auto iNonlinear_Iter = 0ul; iNonlinear_Iter < Nonlinear_Iter; iNonlinear_Iter++) {
    
    /*--- Compute min volume in the entire mesh. ---*/

    ComputeDeforming_Element_Volume(geometry, MinVolume, MaxVolume, Screen_Output);
    if (rank == MASTER_NODE && Screen_Output)
      cout << "Min. volume: " << MinVolume << ", max. volume: " << MaxVolume << "." << endl;

    /*--- Obtaining the interpolation coefficients of the control nodes ---*/
    GetInterpolationCoefficients(geometry, config, iNonlinear_Iter);

    if(1){//TODO -> add option to enable BL preservation
      GetBL_Deformation(geometry, config);
      controlNodes = &boundaryNodes;
      GetInterpolationCoefficients(geometry, config, iNonlinear_Iter);
    }

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

  /*--- Deformation vector only has to be set once ---*/
  if(iNonlinear_Iter == 0){
    SetDeformationVector(geometry, config);
  }
  // for(auto x : *controlNodes){
  //   cout << x->GetIndex() << endl;
  // }
  // for(auto x : deformationVector){
  //   cout << x << endl;
  // }
  /*--- Computing the interpolation matrix with RBF evaluations based on Euclidean distance ---*/
  SetInterpolationMatrix(geometry, config);

  /*--- Solving the RBF system to get the interpolation coefficients ---*/
  SolveRBF_System();   

  cout << "system has been solved " << endl;
}


void CRadialBasisFunctionInterpolation::SetControlNodes(CGeometry* geometry, CConfig* config){
  unsigned short iMarker; 
  unsigned short iVertex; 

  unsigned long nBoundaryLayerNodes = 0, nWallNodes = 0;

  /*--- Total number of boundary nodes (including duplicates of shared boundaries) ---*/
  unsigned long nBoundNodes = 0;
  for(iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++){
    if(!config->GetMarker_All_Deform_Mesh_Internal(iMarker)){
      nBoundNodes += geometry->nVertex[iMarker]; 
    }
    if(1){//TODO if BL preservation
      if(config->GetMarker_All_TagBound(iMarker) == "BOUNDARY_LAYER"){//TODO add function to check if marker is BL marker
        nBoundaryLayerNodes += geometry->nVertex[iMarker];
      }else if(config->GetMarker_All_TagBound(iMarker) == "CYLINDER"){//TODO if marker is wall marker
        nWallNodes += geometry->nVertex[iMarker];
      }
    }
  }

  /*--- Vector with boudary nodes has at most nBoundNodes ---*/
  boundaryNodes.resize(nBoundNodes);
  boundaryLayerNodes.resize(nBoundaryLayerNodes);
  wallNodes.resize(nWallNodes);

  /*--- Storing of the global, marker and vertex indices ---*/
  unsigned long count = 0, ii = 0, jj = 0;
  for(iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++){
    if(!config->GetMarker_All_Deform_Mesh_Internal(iMarker)){
      for(iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++){
      
        boundaryNodes[count++] = new CRadialBasisFunctionNode(geometry, iMarker, iVertex);
      
      if(1){//todo if BL preservation
        if(config->GetMarker_All_TagBound(iMarker) == "BOUNDARY_LAYER"){ //TODO -> if marker is a boundary layer marker
          boundaryLayerNodes[ii++] = boundaryNodes[count-1];
        }else if (config->GetMarker_All_TagBound(iMarker) == "CYLINDER"){//TODO if marker is wall marker
          wallNodes[jj++] = boundaryNodes[count-1];
        } 
      }
      }
    }
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
  for(auto i = 0; i < controlNodes->size(); i++){ //TODO change the way the BL edge nodes are dealt with
    // if(config->GetMarker_All_Moving((*controlNodes)[i]->GetMarker())){

      for(auto iDim = 0; iDim < nDim; iDim++){
        deformationVector[i+iDim*controlNodes->size()] = SU2_TYPE::GetValue(geometry->vertex[(*controlNodes)[i]->GetMarker()][(*controlNodes)[i]->GetVertex()]->GetVarCoord()[iDim] * VarIncrement);
      }
    // }
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
  su2double var_coord[nDim];
  
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
  
  /*--- Applying the surface deformation, which are stored in the deformation vector ---*/
  for(cNode = 0; cNode < controlNodes->size(); cNode++){ //TODO deal differently with BL edge nodes
    // if(config->GetMarker_All_Moving((*controlNodes)[cNode]->GetMarker())){
      for(iDim = 0; iDim < nDim; iDim++){
        geometry->nodes->AddCoord((*controlNodes)[cNode]->GetIndex(), iDim, deformationVector[cNode + iDim*controlNodes->size()]);
      }
    // }
  }  

}

void CRadialBasisFunctionInterpolation::GetBL_Deformation(CGeometry* geometry, CConfig* config){
  cout << "obtaining the deformation of the outer BL edge" << endl;
  unsigned long iNode, jNode; 
  unsigned short iDim;


  su2double bl_thickness = 0.0714; //TODO make function to determine this automatically.

  // finding free deformation of boundary layer edge nodes
  su2matrix<su2double> new_coord(boundaryLayerNodes.size(), nDim);

  for(iNode = 0; iNode < boundaryLayerNodes.size(); iNode++){
    // obtain current coordinate
    auto coord = geometry->nodes->GetCoord(boundaryLayerNodes[iNode]->GetIndex());

    // set new_coord equal to old coordinate
    for( iDim = 0; iDim < nDim; iDim++){
      new_coord[iNode][iDim] = coord[iDim];
    }

    for(jNode = 0; jNode < controlNodes->size(); jNode++){

      auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes)[jNode]->GetIndex()), coord);

      auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));

      // apply free deformation 
      for( iDim = 0; iDim < nDim; iDim++){
        new_coord[iNode][iDim] += rbf*coefficients[jNode + iDim*controlNodes->size()];
      }
    }
  }
  

  // adt variables
  vector<su2double> Coord_bound(nDim * controlNodes->size());
  vector<unsigned long> PointIDs(controlNodes->size());
  unsigned long pointID;  
  su2double dist;
  int rankID;
  unsigned long ii = 0;


  // applying the deformation to the control nodes (wall nodes)
  // and obtaining the information for the ad tree
  for( jNode = 0; jNode < controlNodes->size(); jNode++){
    PointIDs[jNode] = jNode;
    for(iDim = 0; iDim < nDim; iDim++){
      geometry->nodes->AddCoord((*controlNodes)[jNode]->GetIndex(), iDim, deformationVector[jNode + iDim * controlNodes->size()]);
      Coord_bound[ii++] = geometry->nodes->GetCoord((*controlNodes)[jNode]->GetIndex())[iDim];
    }
  }

  // applying the deformation of the control nodes and updating the CV's to obtain new normals 
  geometry->SetBoundControlVolume(config, UPDATE);

  // assembly of the ad tree
  CADTPointsOnlyClass WallADT(nDim, controlNodes->size(), Coord_bound.data(), PointIDs.data(), true);

  su2double nc[nDim];
  su2double dist_vec[nDim];
  su2double added_thickness;
  for(iNode = 0; iNode < boundaryLayerNodes.size(); iNode++){

    WallADT.DetermineNearestNode(new_coord[iNode], dist, pointID, rankID);
    
    auto normal = geometry->vertex[(*controlNodes)[pointID]->GetMarker()][(*controlNodes)[pointID]->GetVertex()]->GetNormal(); 
    auto normal_length = GeometryToolbox::Norm(nDim, normal);

    for(iDim = 0; iDim < nDim; iDim++){
      normal[iDim] = normal[iDim]/normal_length;
    }
    GeometryToolbox::Distance(nDim, new_coord[iNode], geometry->nodes->GetCoord((*controlNodes)[pointID]->GetIndex()), dist_vec);

    auto dp = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

    added_thickness =  abs(dp) - bl_thickness;
    su2double var_coord[nDim];
    for(iDim = 0; iDim < nDim; iDim++){
      new_coord[iNode][iDim] += added_thickness * normal[iDim];
      var_coord[iDim] = new_coord[iNode][iDim] - geometry->nodes->GetCoord(boundaryLayerNodes[iNode]->GetIndex())[iDim];
    }
    geometry->vertex[boundaryLayerNodes[iNode]->GetMarker()][boundaryLayerNodes[iNode]->GetVertex()]->SetVarCoord(var_coord); //TODO include consideration for number of deformation steps
  }

  for( jNode = 0; jNode < controlNodes->size(); jNode++){
    for(iDim = 0; iDim < nDim; iDim++){
      geometry->nodes->AddCoord((*controlNodes)[jNode]->GetIndex(), iDim, -deformationVector[jNode + iDim * controlNodes->size()]);
    }
  }
  // applying the deformation of the control nodes and updating the CV's to obtain new normals 
  geometry->SetBoundControlVolume(config, UPDATE);
  
  // add bl nodes to the control nodes
  for(iNode = 0; iNode < boundaryLayerNodes.size(); iNode++){
    controlNodes->push_back(move(boundaryLayerNodes[iNode]));
  }
}