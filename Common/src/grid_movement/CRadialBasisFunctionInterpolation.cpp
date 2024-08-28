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
  // if(iNonlinear_Iter == 0){
  SetDeformationVector(geometry, config);
  // }

  /*--- Computing the interpolation matrix with RBF evaluations based on Euclidean distance ---*/
  SetInterpolationMatrix(geometry, config);

  /*--- Solving the RBF system to get the interpolation coefficients ---*/
  SolveRBF_System();

  cout << "system has been solved " << endl;
}


void CRadialBasisFunctionInterpolation::SetControlNodes(CGeometry* geometry, CConfig* config){
  unsigned short iMarker, iMarker_Edge = 0, iMarker_Wall = 0; 
  unsigned long iVertex; 


  /*--- Data structure for the inflation layer nodes ---*/
  if(config->GetBL_Preservation()){
    InflationLayer_EdgeNodes = new vector<CRadialBasisFunctionNode*>*[config->GetnMarker_BoundaryLayer()];
    InflationLayer_WallNodes = new vector<CRadialBasisFunctionNode*>*[config->GetnMarker_Wall()];
  }
    
  
  /*--- Storing of the global, marker and vertex indices ---*/
  unsigned long count = 0, ii = 0, jj = 0;
  bool check_bound;
  vector<CRadialBasisFunctionNode*>* push_vector;
  for(iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++){
    cout << config->GetMarker_All_Moving(iMarker) << '\t' << config->GetMarker_All_TagBound(iMarker) << endl;
    
    /*--- */
    if (config->GetMarker_All_Deform_Mesh_Internal(iMarker)){
      cout << "internal marker, continuing to next" << endl;
      continue;
    }

    // if moving marker -> but only if not mesh wall marker -> overwrite
    if(config->GetMarker_All_Moving(iMarker)){
      cout << "moving marker" << endl;
      push_vector = &boundaryNodes;
      check_bound = false;
    }
    
    if (config->GetBL_Preservation()){
      
      if (config->GetMarker_All_Wall(iMarker)){
        InflationLayer_WallNodes[iMarker_Wall] = new vector <CRadialBasisFunctionNode*>;
        push_vector = InflationLayer_WallNodes[iMarker_Wall];
        iMarker_Wall++;
        check_bound = true;
      }

      if (config->GetMarker_All_BoundaryLayer(iMarker)){
        InflationLayer_EdgeNodes[iMarker_Edge] = new vector <CRadialBasisFunctionNode*>;
        push_vector = InflationLayer_EdgeNodes[iMarker_Edge];
        iMarker_Edge++;
        check_bound = true;
      }

    }

    for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++){
      if (check_bound){
        if (geometry->nodes->GetPhysicalBoundary(geometry->vertex[iMarker][iVertex]->GetNode()))
          boundaryNodes.push_back(new CRadialBasisFunctionNode(geometry, iMarker, iVertex));        
        else
          push_vector->push_back(new CRadialBasisFunctionNode(geometry, iMarker, iVertex));
      }
      else
        push_vector->push_back(new CRadialBasisFunctionNode(geometry, iMarker, iVertex));
    }

  }

  
  /*--- Sorting of the boundary nodes based on global index ---*/
  sort(boundaryNodes.begin(), boundaryNodes.end(), Compare);

  /*--- Obtaining unique set of boundary nodes ---*/
  boundaryNodes.resize(std::distance(boundaryNodes.begin(), unique(boundaryNodes.begin(), boundaryNodes.end(), Equal)));


  controlNodes.resize(1); //TODO better placement
  if(!config->GetBL_Preservation()){
    controlNodes[0] = &boundaryNodes; // if case of no preservation of the inflation layer, the control nodes are the boundaryNodes. 
    // in case of inflation layer preservation, the controlNodes are assigned in the GetBL_deformation function
  }

  ofstream file;
  file.open("control_nodes.txt");
  file << "boundary nodes: \n";
  for(auto x : boundaryNodes){
    file << x->GetIndex() << ", ";
  }
  if(config->GetBL_Preservation()){
  file << "\ninflation wall nodes: \n";
  for(auto x : (*InflationLayer_WallNodes[0])){
    file << x->GetIndex() << ", ";
  }
  file << "\ninflation edge nodes: \n";
  for(auto x : (*InflationLayer_EdgeNodes[0])){
    file << x->GetIndex() << ", ";
  }
  }
  file.close();

  //TODO find the total number of inflationLayerWallNodes
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
  for (auto i = 0; i < controlNodes.size(); i++){

    for (auto j = 0; j < controlNodes[i]->size(); j++){

      for (auto iDim = 0; iDim < nDim; iDim++){
        deformationVector[ii+iDim*GetnControlNodes()] = SU2_TYPE::GetValue(geometry->vertex[(*controlNodes[i])[j]->GetMarker()][(*controlNodes[i])[j]->GetVertex()]->GetVarCoord()[iDim] * VarIncrement);   
      }

      ii++;
    }
  }  
}

void CRadialBasisFunctionInterpolation::SetInternalNodes(CGeometry* geometry, CConfig* config){
  unsigned short iMarker, iWallMarker= 0, jNode, iDim;
  unsigned long iNode, iElem; 

  // unique_ptr<CADTElemClass> ElemADT2;
  // if (config->GetBL_Preservation()){
  //   ElemADT2 = GetWallADT(geometry, config);
  // }
  
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
  

    // get number of wall nodes
    unsigned long nWallNodes = 0;//GetnWallVertices(config);

    //loop over all nodes
    for (iNode = 0; iNode < geometry->GetnPoint(); iNode++) {
      // in case of wall node update?
      if (wallVertices[iNode]) {
        wallVertices[iNode] = nWallNodes++;

        for (iDim = 0; iDim < nDim; iDim++) surfaceCoor.push_back(geometry->nodes->GetCoord(iNode, iDim));
      }
    }
    for(iNode = 0; iNode < surfaceConn.size(); iNode++) surfaceConn[iNode] = wallVertices[surfaceConn[iNode]];
    
  }

  CADTElemClass ElemADT(nDim, surfaceCoor, surfaceConn, VTK_TypeElem, markerIDs, elemIDs, true);
  

  /*--- resizing the internal nodes vector ---*/
  internalNodes.resize(2*geometry->GetnPoint());

  // if inflation layer preservation
  InflationLayer_InternalNodes = new vector<unsigned long>*[config->GetnMarker_Wall()];
  for(unsigned short iMarker = 0; iMarker < config->GetnMarker_Wall(); iMarker++){
    InflationLayer_InternalNodes[iMarker] = new vector<unsigned long>;
  }
  

  /*--- Looping over all nodes and check if present in boundary nodes vector ---*/
  unsigned long idx_cnt = 0;  
  unsigned short nearest_marker; unsigned long nearest_elem;

  ofstream file1;

  file1.open("setinternalnodes2.txt");
  
  

  for(iNode = 0; iNode < geometry->GetnPoint(); iNode++){

    // if not part of boundary
    if(!geometry->nodes->GetBoundary(iNode)){
      
      // in case of inflation layer preservation
      if(config->GetBL_Preservation()){
        
      //  ElemADT.DetermineNearestElement(geometry->nodes->GetCoord(iNode), dist, nearest_marker, nearest_elem, rankID);
      //  cout << dist << '\t' << nearest_marker << '\t' << nearest_elem << endl;
       ElemADT.DetermineNearestElement(geometry->nodes->GetCoord(iNode), dist, nearest_marker, nearest_elem, rankID);
       cout << geometry->nodes->GetCoord(iNode)[0] << "\t" << geometry->nodes->GetCoord(iNode)[1] << "\t"<<  dist << "\t" << nearest_marker << "\t" << nearest_elem << endl;

      //  TODO for some reason the distance from ElemADT is not correct. 
      //  auto distance = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord(iNode), geometry->vertex[nearest_marker][nearest_elem]->GetCoord());

      //  cout << "check" << endl;
      //  cout << distance << '\t' << geometry->nodes->GetCoord(iNode)[0] << '\t' << geometry->nodes->GetCoord(iNode)[1] << '\t' << geometry->vertex[nearest_marker][nearest_elem]->GetCoord()[0] << '\t' << geometry->vertex[nearest_marker][nearest_elem]->GetCoord()[1] << endl;
       file1 << dist << '\t' << nearest_marker << '\t' << nearest_elem <<  endl;
       

      

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
  file1.close();
  for(auto iMarker = 0u; iMarker < config->GetnMarker_All();iMarker++){
    if (config->GetMarker_All_Deform_Mesh_Internal(iMarker)){
      for (auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++){
        internalNodes[idx_cnt++] = geometry->vertex[iMarker][iVertex]->GetNode();
      }
    }

    

    
  }

  // setting to actual size
  internalNodes.resize(idx_cnt);

  //TODO remove next print statements
  ofstream file;

  file.open("inodes.txt");
  if(config->GetBL_Preservation()){
  file << "internal inf lay nodes: \n";
  for(auto x : (*InflationLayer_InternalNodes[0])){
    file << x << ", ";
  }
  }
  file << "\ninternal nodes: \n";
  for (auto x : internalNodes){
    file << x << ", ";
  }
  file.close();
  // exit(0);
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
  
  auto Nonlinear_iter = config->GetGridDef_Nonlinear_Iter();
  controlNodes.resize(1);
  // looping over the inflation layer markers
  for(iMarker = 0; iMarker < config->GetnMarker_BoundaryLayer(); iMarker++){

    //assigning the correct control nodes;
    controlNodes[0] = InflationLayer_WallNodes[iMarker];

    // solving the interpolation system to obtain the interpolation coefficients of the control nodes
    GetInterpolationCoefficients(geometry, config, 0); 

    // finding free deformation of boundary layer edge nodes
    su2matrix<su2double> new_coord(InflationLayer_EdgeNodes[iMarker]->size(), nDim); //TODO should be resized instead of reinitialized
    
    // for(auto x = 0; x < InflationLayer_WallNodes[iMarker]->size();x++){
    //   cout << deformationVector[x] << "\t" << coefficients[x] <<  endl;
    // }



    // looping through nodes of the inflation layer edge
    for(iNode = 0; iNode < InflationLayer_EdgeNodes[iMarker]->size(); iNode++){

      // obtain current coordinate
      auto coord = geometry->nodes->GetCoord((*InflationLayer_EdgeNodes[iMarker])[iNode]->GetIndex());

      // set new_coord equal to old coordinate
      for( iDim = 0; iDim < nDim; iDim++){
        new_coord[iNode][iDim] = coord[iDim];
      }
      // cout <<"\n"<< new_coord[iNode][0] << '\t' << new_coord[iNode][1] << '\t' << new_coord[iNode][2] << endl;
      // looping  through inflation layer wall nodes
      for(jNode = 0; jNode < InflationLayer_WallNodes[iMarker]->size(); jNode++){
        // cout << coord << endl;
        auto dist = GeometryToolbox::Distance(nDim, geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[jNode]->GetIndex()), coord);

        auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(kindRBF, radius, dist));

        // apply free deformation 
        for( iDim = 0; iDim < nDim; iDim++){
          new_coord[iNode][iDim] += rbf*coefficients[jNode + iDim*InflationLayer_WallNodes[iMarker]->size()];
        }
        
      }
      
      // cout << new_coord[iNode][0] << '\t' << new_coord[iNode][1] << '\t' << new_coord[iNode][2] << endl;
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
      //determine nearest wall node
      WallADT.DetermineNearestNode(new_coord[iNode], dist, pointID, rankID);
      // cout << "edge node: " << (*InflationLayer_EdgeNodes[iMarker])[iNode]->GetIndex() << endl;
      // cout << "coord: " << new_coord[iNode][0] << '\t' << new_coord[iNode][1] << '\t' << new_coord[iNode][2] << endl;
      //get normal and make it a unit vector
      auto normal = geometry->vertex[(*InflationLayer_WallNodes[iMarker])[pointID]->GetMarker()][(*InflationLayer_WallNodes[iMarker])[pointID]->GetVertex()]->GetNormal(); 
    
      auto normal_length = GeometryToolbox::Norm(nDim, normal);
      
      for(iDim = 0; iDim < nDim; iDim++){
        normal[iDim] = normal[iDim]/normal_length;
      }

      // cout << "closest wall: " << (*InflationLayer_WallNodes[iMarker])[pointID]->GetIndex() << "\t";
      // cout << "coord; " << geometry->vertex[(*InflationLayer_WallNodes[iMarker])[pointID]->GetMarker()][(*InflationLayer_WallNodes[iMarker])[pointID]->GetVertex()]->GetCoord()[0] << '\t' << geometry->vertex[(*InflationLayer_WallNodes[iMarker])[pointID]->GetMarker()][(*InflationLayer_WallNodes[iMarker])[pointID]->GetVertex()]->GetCoord()[1]  << '\t' << geometry->vertex[(*InflationLayer_WallNodes[iMarker])[pointID]->GetMarker()][(*InflationLayer_WallNodes[iMarker])[pointID]->GetVertex()]->GetCoord()[2] << endl;
      // cout << "normal: " << normal[0] << '\t' << normal[1] << '\t' << normal[2] << endl;
      // find distance of freely displaced edge node to nearest wall node. 
      GeometryToolbox::Distance(nDim, new_coord[iNode], geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[pointID]->GetIndex()), dist_vec);
      // cout << "dist_vec: " << dist_vec[0] << '\t' << dist_vec[1] << '\t' <<dist_vec[2] << endl;
      // dot product of normal and distance vector
      auto dp = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

      // required additional inflation layer thickness
      // if( abs(dp) > bl_thickness[iMarker]){
        added_thickness = - bl_thickness[iMarker] + abs(dp); // TODO sign keeps changing somehow (started as +, - for 3D | -, + for 2D)
      // }else added_thi.ckness= 0;
      
      
      // cout << dp << '\t' << added_thickness << endl;
      // cout << bl_thickness[iMarker] << endl;
      

      // required variation in coords to maintain inflation layer height
      su2double var_coord[nDim];
      for(iDim = 0; iDim < nDim; iDim++){
        new_coord[iNode][iDim] += added_thickness * normal[iDim];
        var_coord[iDim] = (new_coord[iNode][iDim] - geometry->nodes->GetCoord((*InflationLayer_EdgeNodes[iMarker])[iNode]->GetIndex())[iDim])*Nonlinear_iter;
      }

      GeometryToolbox::Distance(nDim, new_coord[iNode], geometry->nodes->GetCoord((*InflationLayer_WallNodes[iMarker])[pointID]->GetIndex()), dist_vec);
      dp = GeometryToolbox::DotProduct(nDim, normal, dist_vec);
      // cout << "coord: " << new_coord[iNode][0] << '\t' << new_coord[iNode][1] << '\t' << new_coord[iNode][2] << endl;
      // cout << "check: " << dp-bl_thickness[iMarker] << endl;
      // exit(0);
      // TODO include consideration for number of deformation steps
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
  // exit(0);
}

void CRadialBasisFunctionInterpolation::GetBL_Thickness(CGeometry* geometry, CConfig* config){
  cout << "determining the BL thickness... " << endl;
  bl_thickness = new su2double[config->GetnMarker_Wall()];
  
  unsigned short iMarker;
  unsigned long nWallNodes = GetnWallVertices(config);
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

unsigned long CRadialBasisFunctionInterpolation::GetnWallVertices(CConfig* config){
  unsigned long nVertices = 0;
  for(auto iPointer = 0; iPointer < config->GetnMarker_Wall(); iPointer++){
    nVertices += InflationLayer_WallNodes[iPointer]->size();
  }
  return nVertices;
}


std::unique_ptr<CADTElemClass> CRadialBasisFunctionInterpolation::GetWallADT(CGeometry* geometry,  CConfig* config){
  unsigned short iMarker;
  unsigned long iElem;
  unsigned short iWallMarker = 0;

  unsigned short jNode, iDim;
  unsigned long iNode;

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
  

  
    unsigned long nWallNodes = GetnWallVertices(config);
    for (iNode = 0; iNode < geometry->GetnPoint(); iNode++) {
      if (wallVertices[iNode]) {
        wallVertices[iNode] = nWallNodes++;

        for (iDim = 0; iDim < nDim; iDim++) surfaceCoor.push_back(geometry->nodes->GetCoord(iNode, iDim));
      }
    }
    for(iNode = 0; iNode < surfaceConn.size(); iNode++) surfaceConn[iNode] = wallVertices[surfaceConn[iNode]];
    
  }

  std::unique_ptr<CADTElemClass> ElemADT (new CADTElemClass(nDim, surfaceCoor, surfaceConn, VTK_TypeElem, markerIDs, elemIDs, true));
  
  return ElemADT;

}