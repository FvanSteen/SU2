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

  if(config->GetBL_Preservation()){
    GetBL_Thickness(geometry, config);
  }

  SetInternalNodes(geometry, config);


  
  
  /*--- Looping over the number of deformation iterations ---*/
  for (auto iNonlinear_Iter = 0ul; iNonlinear_Iter < Nonlinear_Iter; iNonlinear_Iter++) {
    
    /*--- Compute min volume in the entire mesh. ---*/

    ComputeDeforming_Element_Volume(geometry, MinVolume, MaxVolume, Screen_Output);
    if (rank == MASTER_NODE && Screen_Output)
      cout << "Min. volume: " << MinVolume << ", max. volume: " << MaxVolume << "." << endl;

    if(config->GetBL_Preservation()){
      GetBL_Deformation(geometry, config);
    }

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

  /*--- Deformation vector only has to be set once ---*/
  if(iNonlinear_Iter == 0){
    SetDeformationVector(geometry, config);
  }

  /*--- Computing the interpolation matrix with RBF evaluations based on Euclidean distance ---*/
  SetInterpolationMatrix(geometry, config);

  /*--- Solving the RBF system to get the interpolation coefficients ---*/
  SolveRBF_System();   

  cout << "system has been solved " << endl;
}


void CRadialBasisFunctionInterpolation::SetControlNodes(CGeometry* geometry, CConfig* config){
  unsigned short iMarker, iMarker_Edge = 0, iMarker_Wall = 0; 
  unsigned long iVertex; 


  // setting size of the arrays containing the pointers to the vectors with CRadialBasisFunctionNode pointers
  if(config->GetBL_Preservation()){
    InflationLayer_EdgeNodes = new vector<CRadialBasisFunctionNode*>*[config->GetnMarker_BoundaryLayer()];
    InflationLayer_WallNodes = new vector<CRadialBasisFunctionNode*>*[config->GetnMarker_Wall()];
  }
  

  /*--- Total number of boundary nodes (including duplicates of shared boundaries) ---*/
  unsigned long nBoundNodes = 0;
  nWallNodes = 0;

  for(iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++){
    if(!config->GetMarker_All_Deform_Mesh_Internal(iMarker) && !config->GetMarker_All_BoundaryLayer(iMarker) && !config->GetMarker_All_Wall(iMarker)){
      nBoundNodes += geometry->nVertex[iMarker];
    }
  }

  /*--- Vector with boundary nodes has at most nBoundNodes ---*/
  boundaryNodes.resize(nBoundNodes);

  
  /*--- Storing of the global, marker and vertex indices ---*/
  unsigned long count = 0, ii = 0, jj = 0;
  for(iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++){
    
    // first test whether it is a boundary that can be ignored
    if(!config->GetMarker_All_Deform_Mesh_Internal(iMarker)){
      
      if(config->GetMarker_All_Wall(iMarker)){ //TODO this if statement is only possible if IL preservation is enabled
        InflationLayer_WallNodes[iMarker_Wall] = new vector <CRadialBasisFunctionNode*>(geometry->nVertex[iMarker]);

        for(iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++){
          (*InflationLayer_WallNodes[iMarker_Wall])[iVertex] = new CRadialBasisFunctionNode(geometry, iMarker, iVertex);
        }

        iMarker_Wall++;
        nWallNodes += geometry->nVertex[iMarker];
      }

      else if (config->GetMarker_All_BoundaryLayer(iMarker)){ //TODO same as above
        InflationLayer_EdgeNodes[iMarker_Edge] = new vector <CRadialBasisFunctionNode*>(geometry->nVertex[iMarker]);
        
        for(iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++){
          (*InflationLayer_EdgeNodes[iMarker_Edge])[iVertex] = new CRadialBasisFunctionNode(geometry, iMarker, iVertex);
        }

        iMarker_Edge++;
      }
      else{
        for(iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++){
          boundaryNodes[count++] = new CRadialBasisFunctionNode(geometry, iMarker, iVertex);  
        }
      }


    }  
  }


  /*--- Sorting of the boundary nodes based on global index ---*/
  sort(boundaryNodes.begin(), boundaryNodes.end(), Compare);

  /*--- Obtaining unique set of boundary nodes ---*/
  boundaryNodes.resize(std::distance(boundaryNodes.begin(), unique(boundaryNodes.begin(), boundaryNodes.end(), Equal)));


  controlNodes.resize(1);
  if(!config->GetBL_Preservation()){
    controlNodes[0] = &boundaryNodes; // if case of no preservation of the inflation layer, the control nodes are the boundaryNodes. 
    // in case of inflation layer preservation, the controlNodes are assigned in the GetBL_deformation function
  }

  //TODO remove next print statements
  // cout << "inflation wall nodes: " << endl;
  // for(auto x : (*InflationLayer_WallNodes[0])){
  //   cout << x->GetIndex() << endl;
  // }

  // cout << "inflation edge nodes: " << endl;
  // for(auto x : (*InflationLayer_EdgeNodes[0])){
  //   cout << x->GetIndex() << endl;
  // }

  // cout << "boundary nodes: " << endl;
  // for(auto x : boundaryNodes){
  //   cout << x->GetIndex() << endl;
  // }

  // exit(0);

  

}

void CRadialBasisFunctionInterpolation::SetInterpolationMatrix(CGeometry* geometry, CConfig* config){
  unsigned long iNode, jNode;
  unsigned short iPtr, jPtr;

  /*--- Initialization of the interpolation matrix ---*/
  interpMat.Initialize(GetnControlNodes());

  /*--- Construction of the interpolation matrix. 
    Since this matrix is symmetric only upper halve has to be considered ---*/
  unsigned long start = 0;
  // blocks on the main diagonal (iPtr == jPtr)
  for(iPtr = 0; iPtr < controlNodes.size(); iPtr++){
    
    /*--- Looping over target nodes ---*/
    for(iNode = 0; iNode < controlNodes[iPtr]->size(); iNode++){

      /*--- Looping over control nodes ---*/
      for(jNode = iNode; jNode < controlNodes[iPtr]->size(); jNode++){

        auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes[iPtr])[iNode]->GetIndex()), geometry->nodes->GetCoord((*controlNodes[iPtr])[jNode]->GetIndex()));

        interpMat(start + iNode, start + jNode) = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));
      }
    }
    start += controlNodes[iPtr]->size();
  }
  
  // blocks off the main diagional
  cout << "OFFDIAGONAL BLOCKS" << endl;
  unsigned long start_row = 0;
  unsigned long start_col;
  for(iPtr = 0; iPtr < controlNodes.size(); iPtr++){
    for(jPtr = iPtr + 1; jPtr < controlNodes.size(); jPtr++){

      start_col = 0;
      for(auto j = 0; j < jPtr; j++){
        start_col += controlNodes[j]->size();
      }

      /*--- Looping over target nodes ---*/
      for(iNode = 0; iNode < controlNodes[iPtr]->size(); iNode++){

        /*--- Looping over control nodes ---*/
        for(jNode = 0; jNode < controlNodes[jPtr]->size(); jNode++){
          auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes[iPtr])[iNode]->GetIndex()), geometry->nodes->GetCoord((*controlNodes[jPtr])[jNode]->GetIndex()));

          interpMat(iNode + start_row , jNode + start_col) = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));
        }
      }
    }
    start_row += controlNodes[iPtr]->size();
  } 


  // for(auto i = 0; i <  GetnControlNodes(); i++){
  //   for(auto j = 0; j < GetnControlNodes(); j++){
  //     cout << interpMat(i,j) << '\t';
  //   }
  //   cout << endl;
  // } 

  // for(iPtr = 0; iPtr < controlNodes.size(); iPtr++){

  //   for(jPtr = 0; jPtr < controlNodes.size(); jPtr++){

  //     /*--- Looping over the target nodes ---*/
  //     for(iNode = 0; iNode < controlNodes[iPtr]->size(); iNode++ ){
  //       /*--- Looping over the control nodes ---*/
  //       for (jNode = iNode; jNode < controlNodes[jPtr]->size(); jNode++){

  //         /*--- Distance between nodes ---*/
  //         auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes[iPtr])[iNode]->GetIndex()), geometry->nodes->GetCoord((*controlNodes[jPtr])[jNode]->GetIndex()));
          
  //         /*--- Evaluation of RBF ---*/
  //         interpMat(iNode, jNode) = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));
  //       }
  //     }
  //   }
  // }

  /*--- Obtaining lower halve using symmetry ---*/
  const bool kernelIsSPD = (kindRBF == RADIAL_BASIS::WENDLAND_C2) || (kindRBF == RADIAL_BASIS::GAUSSIAN) ||
                          (kindRBF == RADIAL_BASIS::INV_MULTI_QUADRIC);
  interpMat.Invert(kernelIsSPD);

}

void CRadialBasisFunctionInterpolation::SetDeformationVector(CGeometry* geometry, CConfig* config){
  /* --- Initialization of the deformation vector ---*/
  deformationVector.resize(GetnControlNodes()*nDim, 0.0);

  /*--- If requested (no by default) impose the surface deflections in
    increments and solve the grid deformation with
    successive small deformations. ---*/
  su2double VarIncrement = 1.0 / ((su2double)config->GetGridDef_Nonlinear_Iter());

  /*--- Setting nonzero displacements of the moving markers ---*/ 
  unsigned long ii = 0;
  for(auto i = 0; i < controlNodes.size(); i++){ //TODO change the way the BL edge nodes are dealt with

    for(auto j = 0; j < controlNodes[i]->size(); j++){
    // if(config->GetMarker_All_Moving((*controlNodes)[i]->GetMarker())){

      for(auto iDim = 0; iDim < nDim; iDim++){
        deformationVector[ii+iDim*GetnControlNodes()] = SU2_TYPE::GetValue(geometry->vertex[(*controlNodes[i])[j]->GetMarker()][(*controlNodes[i])[j]->GetVertex()]->GetVarCoord()[iDim] * VarIncrement);
      }
      ii++;
    }
    // }
  }  
}

void CRadialBasisFunctionInterpolation::SetInternalNodes(CGeometry* geometry, CConfig* config){
  unsigned short iMarker, iWallMarker= 0, jNode, iDim;
  unsigned long iNode, iElem; 
  
  vector<su2double> surfaceCoor;
  vector<unsigned long> surfaceConn;
  vector<unsigned long> elemIDs;
  vector<unsigned short> VTK_TypeElem;
  vector<unsigned short> markerIDs;

  su2double dist;
  int rankID;

  // construction of ADT for elements
  if(config->GetBL_Preservation()){ //TODO make function for this that returns pointer to the ad tree

    vector<unsigned long> wallVertices(geometry->GetnPoint(), 0);

    for(iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++){ // loop through wall node boundaries
      
      if(config->GetMarker_All_Wall(iMarker)){

        for (iElem = 0; iElem < geometry->nElem_Bound[iMarker]; iElem++) {
      
          const unsigned short VTK_Type = geometry->bound[iMarker][iElem]->GetVTK_Type();
          const unsigned short nDOFsPerElem = geometry->bound[iMarker][iElem]->GetnNodes();
          
          markerIDs.push_back(iWallMarker);
          VTK_TypeElem.push_back(VTK_Type);
          elemIDs.push_back(iElem);

          for(jNode = 0; jNode < nDOFsPerElem; jNode++){
            iNode = geometry->bound[iMarker][iElem]->GetNode(jNode);
            wallVertices[iNode] = 1;
            surfaceConn.push_back(geometry->bound[iMarker][iElem]->GetNode(jNode));
          }
        }
        iWallMarker++;
      }
    }
  

  
    unsigned long nWallNodes = 0;
    for (iNode = 0; iNode < geometry->GetnPoint(); iNode++) {
      if (wallVertices[iNode]) {
        wallVertices[iNode] = nWallNodes++;

        for (iDim = 0; iDim < nDim; iDim++) surfaceCoor.push_back(geometry->nodes->GetCoord(iNode, iDim));
      }
    }
    for(iNode = 0; iNode < surfaceConn.size(); iNode++) surfaceConn[iNode] = wallVertices[surfaceConn[iNode]];
    
  }

  CADTElemClass ElemADT(nDim, surfaceCoor, surfaceConn, VTK_TypeElem, markerIDs, elemIDs, true);
  

  /*--- resizing the internal nodes vector ---*/
  internalNodes.resize(geometry->GetnPoint());

  // if inflation layer preservation
  InflationLayer_InternalNodes = new vector<unsigned long>*[config->GetnMarker_Wall()];
  for(unsigned short iMarker = 0; iMarker < config->GetnMarker_Wall(); iMarker++){
    InflationLayer_InternalNodes[iMarker] = new vector<unsigned long>;
  }
  

  /*--- Looping over all nodes and check if present in boundary nodes vector ---*/
  unsigned long idx_cnt = 0;  
  unsigned short nearest_marker; unsigned long nearest_elem;

  for(iNode = 0; iNode < geometry->GetnPoint(); iNode++){

    // if not part of boundary
    if(!geometry->nodes->GetBoundary(iNode)){
      
      // in case of inflation layer preservation
      if(config->GetBL_Preservation()){
        
       ElemADT.DetermineNearestElement(geometry->nodes->GetCoord(iNode), dist, nearest_marker, nearest_elem, rankID);
       
       if(dist <= bl_thickness[nearest_marker] ){
        InflationLayer_InternalNodes[nearest_marker]->push_back(iNode);
       }else{
        internalNodes[idx_cnt++] = iNode;
       }
      }else{
        internalNodes[idx_cnt++] = iNode;
      }
    }
  }



  // setting to actual size
  internalNodes.resize(idx_cnt);

  //TODO remove next print statements
  // ofstream file;
  // file.open("inodes2.txt");
  // file << "internal inf lay nodes: \n";
  // for(auto x : (*InflationLayer_InternalNodes[0])){
  //   file << x << ", ";
  // }

  // file << "\ninternal nodes: \n";
  // for (auto x : internalNodes){
  //   file << x << ", ";
  // }


}


void CRadialBasisFunctionInterpolation::SolveRBF_System(){

  /*--- resizing the interpolation coefficient vector ---*/
  coefficients.resize(nDim*GetnControlNodes());

  /*--- Looping through the dimensions in order to find the interpolation coefficients for each direction ---*/
  unsigned short iDim;
  for(iDim = 0; iDim < nDim; iDim++){
    interpMat.MatVecMult(deformationVector.begin()+iDim*GetnControlNodes(), coefficients.begin()+iDim*GetnControlNodes());
  }

  cout << endl;
}

void CRadialBasisFunctionInterpolation::UpdateGridCoord(CGeometry* geometry, CConfig* config){
  cout << "updating the grid coordinates" << endl;
  unsigned long iNode, cNode;
  unsigned short iDim, cPtr;
  
  /*--- Vector for storing the coordinate variation ---*/
  su2double var_coord[nDim];
  unsigned long start;
  /*--- Loop over the internal nodes ---*/



  
  for(iNode = 0; iNode < internalNodes.size(); iNode++){
  // for(iNode = 0; iNode < InflationLayer_InternalNodes[0]->size() ; iNode++){
    start = 0;
    for(cPtr = 0; cPtr < controlNodes.size(); cPtr++){

      /*--- Loop for contribution of each control node ---*/
      for(cNode = 0; cNode < controlNodes[cPtr]->size(); cNode++){
        // cout << "internal: " << internalNodes[iNode] << ", control:" << (*controlNodes[cPtr])[cNode]->GetIndex() << endl;
        /*--- Determine distance between considered internal and control node ---*/
        auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes[cPtr])[cNode]->GetIndex()), geometry->nodes->GetCoord(internalNodes[iNode]));
        // auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes[cPtr])[cNode]->GetIndex()), geometry->nodes->GetCoord((*InflationLayer_InternalNodes[0])[iNode]));
        
        /*--- Evaluate RBF based on distance ---*/
        auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));
        // cout << "distance: " << dist << ", rbf val: " << rbf << endl;
        // cout << "index: " << start+cNode << endl; 
        /*--- Add contribution to total coordinate variation ---*/
        for( iDim = 0; iDim < nDim; iDim++){
          var_coord[iDim] += rbf*coefficients[start + cNode + iDim*GetnControlNodes()];
        }
      }
      start += controlNodes[cPtr]->size();  
    }

    
    /*--- Apply the coordinate variation and resetting the var_coord vector to zero ---*/
    for(iDim = 0; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(internalNodes[iNode], iDim, var_coord[iDim]);
      // geometry->nodes->AddCoord((*InflationLayer_InternalNodes[0])[iNode], iDim, var_coord[iDim]);  
      var_coord[iDim] = 0;
    } 
    
    
  }  
  

  /*--- Applying the surface deformation, which are stored in the deformation vector ---*/
  start = 0;
  for(cPtr = 0; cPtr < controlNodes.size(); cPtr++){
    for(cNode = 0; cNode < controlNodes[cPtr]->size(); cNode++){ //TODO deal differently with BL edge nodes
      // if(config->GetMarker_All_Moving((*controlNodes)[cNode]->GetMarker())){
        for(iDim = 0; iDim < nDim; iDim++){
          geometry->nodes->AddCoord((*controlNodes[cPtr])[cNode]->GetIndex(), iDim, deformationVector[start + cNode + iDim*GetnControlNodes()]);
        }
      // }
    }  
    start += controlNodes[cPtr]->size();
  }
  
}

void CRadialBasisFunctionInterpolation::GetBL_Deformation(CGeometry* geometry, CConfig* config){
  cout << "obtaining the deformation of the outer BL edge" << endl;

  unsigned long iNode, jNode; 
  unsigned short iDim, iMarker;
 

  // looping over the inflation layer markers
  for(iMarker = 0; iMarker < config->GetnMarker_BoundaryLayer(); iMarker++){

    //assigning the correct control nodes;
    controlNodes[0] = InflationLayer_WallNodes[iMarker];

    // solving the interpolation system to obtain the interpolation coefficients of the control nodes
    GetInterpolationCoefficients(geometry, config, 0); //TODO
  
    // finding free deformation of boundary layer edge nodes
    su2matrix<su2double> new_coord(InflationLayer_EdgeNodes[iMarker]->size(), nDim); //TODO should be resized instead of reinitialized

    // looping through nodes of the inflation layer edge
    for(iNode = 0; iNode < InflationLayer_EdgeNodes[iMarker]->size(); iNode++){

      // obtain current coordinate
      auto coord = geometry->nodes->GetCoord((*InflationLayer_EdgeNodes[iMarker])[iNode]->GetIndex());
      
      // set new_coord equal to old coordinate
      for( iDim = 0; iDim < nDim; iDim++){
        new_coord[iNode][iDim] = coord[iDim];
      }
      
      // looping through inflation layer wall nodes
      for(jNode = 0; jNode < InflationLayer_WallNodes[iMarker]->size(); jNode++){

        auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[jNode]->GetIndex()), coord);

        auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));

        // apply free deformation 
        for( iDim = 0; iDim < nDim; iDim++){
          new_coord[iNode][iDim] += rbf*coefficients[jNode + iDim*InflationLayer_WallNodes[iMarker]->size()];
        }
      }
    }
    
    // adt variables for the tree containing the displaced inflation layer wall nodes
    vector<su2double> Coord_bound(nDim * InflationLayer_WallNodes[iMarker]->size());
    vector<unsigned long> PointIDs(InflationLayer_WallNodes[iMarker]->size());
    unsigned long pointID;  
    su2double dist;
    int rankID;
    unsigned long ii = 0;


    // applying the deformation to the control nodes (wall nodes)
    // and obtaining the information for the ad tree
    for( jNode = 0; jNode < InflationLayer_WallNodes[iMarker]->size(); jNode++){
      PointIDs[jNode] = jNode;
      for(iDim = 0; iDim < nDim; iDim++){
        geometry->nodes->AddCoord((*InflationLayer_WallNodes[iMarker])[jNode]->GetIndex(), iDim, deformationVector[jNode + iDim * InflationLayer_WallNodes[iMarker]->size()]);
        Coord_bound[ii++] = geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[jNode]->GetIndex())[iDim];
      }
    }

    // applying the deformation of the control nodes and updating the CV's to obtain new normals 
    geometry->SetBoundControlVolume(config, UPDATE);

    // assembly of the ad tree
    CADTPointsOnlyClass WallADT(nDim, InflationLayer_WallNodes[iMarker]->size(), Coord_bound.data(), PointIDs.data(), true);

    su2double dist_vec[nDim];
    su2double added_thickness;


    //looping through inflation layer edge nodes
    for(iNode = 0; iNode < InflationLayer_EdgeNodes[iMarker]->size(); iNode++){
      cout << (*InflationLayer_EdgeNodes[iMarker])[iNode]->GetIndex() << endl;
      cout << "new coord: " << new_coord[iDim][0] << '\t' << new_coord[iDim][1] << "\t" <<  new_coord[iDim][2]  << endl;
      //determine nearest wall node
      WallADT.DetermineNearestNode(new_coord[iNode], dist, pointID, rankID);
      cout << "nearest node: " << (*InflationLayer_WallNodes[iMarker])[pointID]->GetIndex() << endl;
      cout << "coord: " << geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[pointID]->GetIndex())[0] << '\t' << geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[pointID]->GetIndex())[1] << "\t" << geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[pointID]->GetIndex())[2] <<  endl;
      //get normal and make it a unit vector
      auto normal = geometry->vertex[(*InflationLayer_WallNodes[iMarker])[pointID]->GetMarker()][(*InflationLayer_WallNodes[iMarker])[pointID]->GetVertex()]->GetNormal(); 
    
      auto normal_length = GeometryToolbox::Norm(nDim, normal);

      cout << "normal: ";
      for(iDim = 0; iDim < nDim; iDim++){
        normal[iDim] = normal[iDim]/normal_length;
        cout << normal[iDim] << '\t';
      }

      cout << endl;

      // find distance of freely displaced edge node to nearest wall node. 
      GeometryToolbox::Distance(nDim, new_coord[iNode], geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[pointID]->GetIndex()), dist_vec);
      cout << dist_vec[0] << '\t' << dist_vec[1] << '\t' << dist_vec[2] << endl;
      // dot product of normal and distance vector
      auto dp = GeometryToolbox::DotProduct(nDim, normal, dist_vec);
      cout << "dot product: " << dp << endl;
      // required additional inflation layer thickness
      added_thickness =  bl_thickness[iMarker] - dp;
      cout << "required bl thickness: " << bl_thickness[iMarker] << endl; 
      cout << "added thickness: " << added_thickness << endl;
      // required variation in coords to maintain inflation layer height
      su2double var_coord[nDim];
      for(iDim = 0; iDim < nDim; iDim++){
        new_coord[iNode][iDim] += added_thickness * normal[iDim];
        var_coord[iDim] = new_coord[iNode][iDim] - geometry->nodes->GetCoord((*InflationLayer_EdgeNodes[iMarker])[iNode]->GetIndex())[iDim];
      }
      cout << "updated coord: " << new_coord[iNode][0] << '\t' << new_coord[iNode][1] << "\t" << new_coord[iNode][2] << endl;
      cout << "var coord: " << var_coord[0] << '\t' << var_coord[1] << "\t" << var_coord[2] << endl;

      GeometryToolbox::Distance(nDim, new_coord[iNode], geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[pointID]->GetIndex()), dist_vec);
      dp = GeometryToolbox::DotProduct(nDim, normal, dist_vec);
      cout << "check dot product: " << dp << endl;
      //TODO include consideration for number of deformation steps
      geometry->vertex[(*InflationLayer_EdgeNodes[iMarker])[iNode]->GetMarker()][(*InflationLayer_EdgeNodes[iMarker])[iNode]->GetVertex()]->SetVarCoord(var_coord); 

    }

    

    // set wall nodes back to initial position //TODO check if this displacement actually has to be performed or that some replacement can take place
    for( jNode = 0; jNode < InflationLayer_WallNodes[iMarker]->size(); jNode++){
      for(iDim = 0; iDim < nDim; iDim++){
        geometry->nodes->AddCoord((*InflationLayer_WallNodes[iMarker])[jNode]->GetIndex(), iDim, -deformationVector[jNode + iDim * InflationLayer_WallNodes[iMarker]->size()]);
      }
    } 

  
    // Setting the right control nodes
    controlNodes.resize(2);
    controlNodes[0] = InflationLayer_WallNodes[iMarker];
    controlNodes[1] = InflationLayer_EdgeNodes[iMarker];

    // solve for the entire displacement of the inflation layer
    GetInterpolationCoefficients(geometry, config, 0);
    

    // updating the inflation layer nodes and wall nodes;
    UpdateInflationLayerCoords(geometry, iMarker);
    
  }

  controlNodes.resize(config->GetnMarker_Wall()+1);
  controlNodes[0] = &boundaryNodes;
  for(iMarker = 0; iMarker < config->GetnMarker_Wall(); iMarker++){
    controlNodes[iMarker+1] = InflationLayer_EdgeNodes[iMarker];
  }
  



  //update the grid
  geometry->SetBoundControlVolume(config, UPDATE);
  // controlNodes = &boundaryNodes;//TODO  
}

void CRadialBasisFunctionInterpolation::GetBL_Thickness(CGeometry* geometry, CConfig* config){
  cout << "determining the BL thickness... " << endl;
  bl_thickness = new su2double[config->GetnMarker_Wall()];
  
  unsigned short iMarker;
  
  vector<su2double> Coord_bound(nDim * nWallNodes);
  vector<unsigned long> PointIDs(nWallNodes);  
  unsigned long pointID;
  su2double dist;
  int rankID;
  unsigned long ii = 0, jj = 0;
  unsigned long jNode;
  unsigned short iDim;
  // applying the deformation to the control nodes (wall nodes)
  // and obtaining the information for the ad tree
  for(iMarker = 0; iMarker < config->GetnMarker_Wall(); iMarker++){
    for(jNode = 0; jNode < InflationLayer_WallNodes[iMarker]->size(); jNode++){
      PointIDs[jj] = jj;
      jj++;
      for(iDim = 0; iDim < nDim; iDim++){
        Coord_bound[ii++] = geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[jNode]->GetIndex())[iDim];
      }
    }
  }

  // assembly of the ad tree
  CADTPointsOnlyClass WallADT(nDim, nWallNodes, Coord_bound.data(), PointIDs.data(), true);

  unsigned short iThickness;
  for(iThickness = 0; iThickness < config->GetnMarker_Wall(); iThickness++){
    WallADT.DetermineNearestNode(geometry->nodes->GetCoord((*InflationLayer_EdgeNodes[iThickness])[0]->GetIndex()), dist, pointID, rankID);
    bl_thickness[iThickness] = dist;
  }
}

unsigned long CRadialBasisFunctionInterpolation::GetnControlNodes(){
  unsigned long nControlNodes = 0;
  for(auto iPointer = 0; iPointer < controlNodes.size(); iPointer++){
    nControlNodes += controlNodes[iPointer]->size();
  }
  return nControlNodes;
}


void CRadialBasisFunctionInterpolation::UpdateInflationLayerCoords(CGeometry* geometry, unsigned short iMarker){
  cout << "updating the inflation layer coordinates" << endl;
  unsigned long iNode, cNode, start;
  unsigned short cPtr, iDim;

  su2double var_coord[nDim];


  for(iNode = 0; iNode < InflationLayer_InternalNodes[iMarker]->size() ; iNode++){
    start = 0;
    for(cPtr = 0; cPtr < controlNodes.size(); cPtr++){

      /*--- Loop for contribution of each control node ---*/
      for(cNode = 0; cNode < controlNodes[cPtr]->size(); cNode++){
        // cout << "internal: " << internalNodes[iNode] << ", control:" << (*controlNodes[cPtr])[cNode]->GetIndex() << endl;
        /*--- Determine distance between considered internal and control node ---*/
        // auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes[cPtr])[cNode]->GetIndex()), geometry->nodes->GetCoord(internalNodes[iNode]));
        auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*controlNodes[cPtr])[cNode]->GetIndex()), geometry->nodes->GetCoord((*InflationLayer_InternalNodes[iMarker])[iNode]));
        
        /*--- Evaluate RBF based on distance ---*/
        auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));
        // cout << "distance: " << dist << ", rbf val: " << rbf << endl;
        // cout << "index: " << start+cNode << endl; 
        /*--- Add contribution to total coordinate variation ---*/
        for( iDim = 0; iDim < nDim; iDim++){
          var_coord[iDim] += rbf*coefficients[start + cNode + iDim*GetnControlNodes()];
        }
      }
      start += controlNodes[cPtr]->size();  
    }

    
    /*--- Apply the coordinate variation and resetting the var_coord vector to zero ---*/
    for(iDim = 0; iDim < nDim; iDim++){
      // geometry->nodes->AddCoord(internalNodes[iNode], iDim, var_coord[iDim]);
      geometry->nodes->AddCoord((*InflationLayer_InternalNodes[iMarker])[iNode], iDim, var_coord[iDim]);  
      var_coord[iDim] = 0;
    } 
  }  

  /*--- Applying the surface deformation, which are stored in the deformation vector ---*/  
  for(cNode = 0; cNode < controlNodes[0]->size(); cNode++){ //TODO deal differently with BL edge nodes
    // if(config->GetMarker_All_Moving((*controlNodes)[cNode]->GetMarker())){
      for(iDim = 0; iDim < nDim; iDim++){
        geometry->nodes->AddCoord((*controlNodes[0])[cNode]->GetIndex(), iDim, deformationVector[cNode + iDim*GetnControlNodes()]);
      }
    // }
  }  
}

