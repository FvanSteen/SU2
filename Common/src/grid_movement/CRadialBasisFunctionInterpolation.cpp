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


CRadialBasisFunctionInterpolation::CRadialBasisFunctionInterpolation(CGeometry* geometry, CConfig* config) : CVolumetricMovement(geometry) {}

CRadialBasisFunctionInterpolation::~CRadialBasisFunctionInterpolation() = default;

void CRadialBasisFunctionInterpolation::SetVolume_Deformation(CGeometry* geometry, CConfig* config, bool UpdateGeo, bool Derivative,
                                                bool ForwardProjectionDerivative){
  // #ifdef HAVE_MPI
  //   // if(rank == 5){
  //     {
  //       volatile int i = 0;
  //       char hostname[256];
  //       gethostname(hostname, sizeof(hostname));
  //       printf("PID %d on %s ready for attach\n", getpid(), hostname);
  //       fflush(stdout); 
  //       while (0 == i)
  //           sleep(2);
  //     }
  //   // }
  // #endif

  /*--- Retrieve type of RBF and its support radius ---*/ 

  const auto kindRBF = config->GetKindRadialBasisFunction();
  const su2double radius = config->GetRadialBasisFunctionParameter();
  
  su2double MinVolume, MaxVolume;

  /*--- Retrieving number of deformation steps and screen output from config ---*/

  const auto Nonlinear_Iter = config->GetGridDef_Nonlinear_Iter();
  auto Screen_Output = config->GetDeform_Output();
  
  /*--- Disable the screen output if we're running SU2_CFD ---*/

  if (config->GetKind_SU2() == SU2_COMPONENT::SU2_CFD && !Derivative) Screen_Output = false;
  if (config->GetSmoothGradient()) Screen_Output = true;


  /*--- Determining the boundary and internal nodes. Setting the control nodes. ---*/ 
  SetBoundNodes(geometry, config);
  
  vector<unsigned long> internalNodes; 
  SetInternalNodes(geometry, config, internalNodes); 
  
  SetCtrlNodes(config);

  /*--- Looping over the number of deformation iterations ---*/
  for (auto iNonlinear_Iter = 0ul; iNonlinear_Iter < Nonlinear_Iter; iNonlinear_Iter++) {
    
    /*--- Compute min volume in the entire mesh. ---*/

    ComputeDeforming_Element_Volume(geometry, MinVolume, MaxVolume, Screen_Output);

    if (rank == MASTER_NODE && Screen_Output)
      cout << "Min. volume: " << MinVolume << ", max. volume: " << MaxVolume << "." << endl;
    

    /*--- Solving the RBF system, resulting in the interpolation coefficients ---*/
    SolveRBF_System(geometry, config, kindRBF, radius);


    /*--- Updating the coordinates of the grid ---*/
    UpdateGridCoord(geometry, config, kindRBF, radius, internalNodes);
    
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

void CRadialBasisFunctionInterpolation::SolveRBF_System(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius){
  
  /*--- In case of data reduction an iterative greedy algorithm is applied 
          to perform the interpolation with a reduced set of control nodes.
          Otherwise with a full set of control nodes. ---*/
  
  if(config->GetRBF_DataReduction()){
    
    /*--- Local maximum error node and corresponding maximum error  ---*/
    vector <unsigned long> maxErrorNodes;

    su2double maxErrorLocal{0};
    
    /*--- Obtaining the initial maximum error nodes, which are found based on the maximum applied deformation.
              Determining the data reduction tolerance, which is equal to a specified factor of the maximum total deformation. ---*/
    if(nCtrlNodesGlobal == 0){
      GetInitMaxErrorNode(geometry, config, maxErrorNodes, maxErrorLocal); 
      SU2_MPI::Allreduce(&maxErrorLocal, &MaxErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());

      /*--- Error tolerance for the data reduction tolerance ---*/
      DataReductionTolerance = MaxErrorGlobal*config->GetRBF_DataRedTolerance() * ((su2double)config->GetGridDef_Nonlinear_Iter());
    }

    /*--- Number of greedy iterations. ---*/
    unsigned short greedyIter = 0;

    /*--- While the maximum error is above the tolerance, data reduction algorithm is continued. ---*/
    while(MaxErrorGlobal > DataReductionTolerance || greedyIter == 0){ 
      
      /*--- In case of a nonzero local error, control nodes are added ---*/
      if(maxErrorLocal> 0){
        AddControlNode(maxErrorNodes);
      }

      /*--- Obtaining the global number of control nodes. ---*/
      Get_nCtrlNodes();

      /*--- Obtaining the interpolation coefficients. ---*/
      GetInterpCoeffs(geometry, config, type, radius);

      /*--- Determining the interpolation error, of the non-control boundary nodes. ---*/
      GetInterpError(geometry, config, type, radius, maxErrorLocal, maxErrorNodes);       
      SU2_MPI::Allreduce(&maxErrorLocal, &MaxErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());

      if(rank == MASTER_NODE) cout << "Greedy iteration: " << greedyIter << ". Max error: " << MaxErrorGlobal << ". Tol: " << DataReductionTolerance << ". Global nr. of ctrl nodes: "  << nCtrlNodesGlobal << "\n" << endl;
      greedyIter++;

    }  
  }else{
    ControlNodes.resize(1);
    ControlNodes[0] = &IL_WallNodes;
    Get_nCtrlNodes();
    /*--- First deforming the inflation layer ---*/
    if(config->GetRBF_IL_Preservation()){
      GetIL_Deformation(geometry, config, type, radius);
    }

    /*--- Obtaining the interpolation coefficients. ---*/
    GetInterpCoeffs(geometry, config, type, radius);
  }
}

void CRadialBasisFunctionInterpolation::GetInterpCoeffs(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius){
  
  /*--- Obtaining the control nodes coordinates and distributing over all processes. ---*/
  SetCtrlNodeCoords(geometry);

  /*--- Obtaining the deformation of the control nodes. ---*/
  SetDeformation(geometry, config);

  /*--- Computation of the (inverse) interpolation matrix. ---*/
  su2passivematrix invInterpMat;
  ComputeInterpolationMatrix(geometry, type, radius, invInterpMat);

  /*--- Obtaining the interpolation coefficients. ---*/
  ComputeInterpCoeffs(invInterpMat);
}


void CRadialBasisFunctionInterpolation::SetBoundNodes(CGeometry* geometry, CConfig* config){
  
  /*--- Storing of the local node, marker and vertex information of the boundary nodes ---*/

  /*--- Looping over the markers ---*/
  for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++) {

    /*--- Checking if not internal or send/receive marker ---*/
    if (!config->GetMarker_All_Deform_Mesh_Internal(iMarker) && !config->GetMarker_All_SendRecv(iMarker) && !config->GetMarker_All_Deform_Mesh_IL_Wall(iMarker)) {

      /*--- Looping over the vertices of marker ---*/
      for (auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Node in consideration ---*/
        auto iNode = geometry->vertex[iMarker][iVertex]->GetNode();

        /*--- Check whether node is part of the subdomain and not shared with a receiving marker (for parallel computation)*/
        if (geometry->nodes->GetDomain(iNode)) {
          BoundNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));        
        }        
      }
    }

    if(config->GetMarker_All_Deform_Mesh_IL_Wall(iMarker)){
      for ( auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++){
        auto iNode =  geometry->vertex[iMarker][iVertex]->GetNode();
        IL_WallNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
      }
    }
  }

  /*--- Sorting of the boundary nodes based on their index ---*/
  sort(BoundNodes.begin(), BoundNodes.end(), HasSmallerIndex);

  /*--- Obtaining unique set ---*/
  BoundNodes.resize(std::distance(BoundNodes.begin(), unique(BoundNodes.begin(), BoundNodes.end(), HasEqualIndex)));

  ofstream of;
  of.open("boundnodes.txt");
  for(auto x : BoundNodes){ of << x->GetIndex() << endl;}
  of.close();

  of.open("wallnodes.txt");
  for(auto x : IL_WallNodes){ of << x->GetIndex() << endl;}
  of.close();


}

void CRadialBasisFunctionInterpolation::SetCtrlNodes(CConfig* config){
  
  ControlNodes.resize(1);

  /*--- Assigning the control nodes based on whether data reduction is applied or not. ---*/
  if(config->GetRBF_DataReduction()){

    /*--- Control nodes are an empty set ---*/
    ControlNodes[0] = &ReducedControlNodes;

  }else{
    /*--- In case of inflation layer preservation ---*/    
    if(config->GetRBF_IL_Preservation()){

      /*--- Control nodes are the inflation layer wall nodes*/
      ControlNodes[0] = &IL_WallNodes;

    }else{
      /*--- Control nodes are the boundary nodes ---*/
      ControlNodes[0] = &BoundNodes;
    }
  }

  /*--- Obtaining the total number of control nodes. ---*/
  Get_nCtrlNodes();

};

void CRadialBasisFunctionInterpolation::ComputeInterpolationMatrix(CGeometry* geometry, const RADIAL_BASIS& type, const su2double radius, su2passivematrix& invInterpMat){

  /*--- In case of parallel computation, the interpolation coefficients are computed on the master node ---*/

  if(rank == MASTER_NODE){
    CSymmetricMatrix interpMat;

    /*--- Initialization of the interpolation matrix ---*/
    interpMat.Initialize(nCtrlNodesGlobal);

    /*--- Construction of the interpolation matrix. 
      Since this matrix is symmetric only upper halve has to be considered ---*/

  
    /*--- Looping over the target nodes ---*/
    for( auto iNode = 0ul; iNode < nCtrlNodesGlobal; iNode++ ){

      /*--- Looping over the control nodes ---*/
      for ( auto jNode = iNode; jNode < nCtrlNodesGlobal; jNode++){
        
        /*--- Distance between nodes ---*/
        auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[iNode*nDim], CtrlCoords[jNode*nDim]);   

        /*--- Evaluation of RBF ---*/
        interpMat(iNode, jNode) = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));
      }
    }

    /*--- Obtaining lower halve using symmetry ---*/
    const bool kernelIsSPD = (type == RADIAL_BASIS::WENDLAND_C2) || (type == RADIAL_BASIS::GAUSSIAN) ||
                            (type == RADIAL_BASIS::INV_MULTI_QUADRIC);

    /*--- inverting the interpolation matrix ---*/
    interpMat.Invert(kernelIsSPD);
    invInterpMat = interpMat.StealData();
  }
}

void CRadialBasisFunctionInterpolation::SetDeformation(CGeometry* geometry, CConfig* config){

  /* --- Initialization of the deformation vector ---*/
  CtrlNodeDeformation.resize(nCtrlNodesLocal * nDim, 0.0); 

  /*--- If requested (no by default) impose the surface deflections in
    increments and solve the grid deformation with
    successive small deformations. ---*/
  const su2double VarIncrement = 1.0 / ((su2double)config->GetGridDef_Nonlinear_Iter());

  unsigned long idx = 0;

  for(auto iControlNodes : ControlNodes){
    /*--- Loop over the control nodes ---*/
    for (auto iNode = 0ul; iNode < iControlNodes->size(); iNode++) {
      
      /*--- Obtaining displacement of ctrl node ---*/
      su2double* var_coord;

      /*--- Node is an inflation layer edge node if not part of any boundary, 
              its displacement is stored as a member of CRadialBasisFunctionNode. ---*/
      if(geometry->nodes->GetBoundary((*iControlNodes)[iNode]->GetIndex())){
        
        var_coord = geometry->vertex[(*iControlNodes)[iNode]->GetMarker()][(*iControlNodes)[iNode]->GetVertex()]->GetVarCoord();
     
      }else{

        var_coord = (*iControlNodes)[iNode]->GetVarCoord();
      }

      /*--- Setting displacement of the control nodes ---*/
      for ( auto iDim = 0u; iDim < nDim; iDim++ ){
        CtrlNodeDeformation[idx++] = var_coord[iDim] * VarIncrement;
      }  
    }
  }
  
  /*--- In case of a parallel computation, the deformation of all control nodes is send to the master process ---*/
  #ifdef HAVE_MPI
    
    /*--- Array containing the local number of control nodes ---*/
    unsigned long Local_nControlNodesArr[size];

    /*--- gathering local control node coordinate sizes on all processes. ---*/
    SU2_MPI::Allgather(&nCtrlNodesLocal, 1, MPI_UNSIGNED_LONG, Local_nControlNodesArr, 1, MPI_UNSIGNED_LONG, SU2_MPI::GetComm()); 

    /*--- Gathering all deformation vectors on the master node ---*/
    if(rank==MASTER_NODE){

      /*--- resizing the global deformation vector ---*/
      CtrlNodeDeformation.resize(nCtrlNodesGlobal*nDim);

      /*--- Receiving the local deformation vector from other processes ---*/
      unsigned long start_idx = 0;
      for (auto iProc = 0; iProc < size; iProc++) {
        if (iProc != MASTER_NODE) {
          SU2_MPI::Recv(&CtrlNodeDeformation[0] + start_idx, Local_nControlNodesArr[iProc]*nDim, MPI_DOUBLE, iProc, 0, SU2_MPI::GetComm(), MPI_STATUS_IGNORE); 
        }
        start_idx += Local_nControlNodesArr[iProc]*nDim;
      }      

    }else{

      /*--- Sending the local deformation vector to the master node ---*/
      SU2_MPI::Send(CtrlNodeDeformation.data(), nCtrlNodesLocal*nDim, MPI_DOUBLE, MASTER_NODE, 0, SU2_MPI::GetComm());  
    } 
  #endif   
}

void CRadialBasisFunctionInterpolation::SetInternalNodes(CGeometry* geometry, CConfig* config, vector<unsigned long>& internalNodes){ 

  if(config->GetRBF_IL_Preservation()){
    //TODO add as config options
    su2double IL_height = 0.0714027493868399;

    /*--- ADT for wall elements ---*/ 
    vector<su2double> surfaceCoor;
    vector<unsigned long> surfaceConn;
    vector<unsigned long> elemIDs;
    vector<unsigned short> VTK_TypeElem;
    vector<unsigned short> markerIDs;

    su2double dist;
    int rankID;


    unsigned short iWallMarker = 0;
    // construction of ADT for elements
    if(config->GetRBF_IL_Preservation()){ //TODO make function for this that returns pointer to the ad tree

      IL_internalNodes = new vector<unsigned long>*[config->GetnMarker_Deform_Mesh_IL_Wall()];
      for( auto iMarker = 0u; iMarker < config->GetnMarker_Deform_Mesh_IL_Wall(); iMarker++){
        IL_internalNodes[iMarker] = new vector<unsigned long>;
      }

      vector<unsigned long> wallVertices(geometry->GetnPoint(), 0);

      for(auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++){ // loop through wall node boundaries
        
        if(config->GetMarker_All_Deform_Mesh_IL_Wall(iMarker)){

          for (auto iElem = 0ul; iElem < geometry->nElem_Bound[iMarker]; iElem++) {
        
            const unsigned short VTK_Type = geometry->bound[iMarker][iElem]->GetVTK_Type();
            const unsigned short nDOFsPerElem = geometry->bound[iMarker][iElem]->GetnNodes();
            
            markerIDs.push_back(iWallMarker);
            VTK_TypeElem.push_back(VTK_Type);
            elemIDs.push_back(iElem);

            for(auto jNode = 0u; jNode < nDOFsPerElem; jNode++){
              auto iNode = geometry->bound[iMarker][iElem]->GetNode(jNode);
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
      for (auto iNode = 0ul; iNode < geometry->GetnPoint(); iNode++) {
        // in case of wall node update?
        if (wallVertices[iNode]) {
          wallVertices[iNode] = nWallNodes++;

          for (auto iDim = 0u; iDim < nDim; iDim++) surfaceCoor.push_back(geometry->nodes->GetCoord(iNode, iDim));
        }
      }
      for(auto iNode = 0u; iNode < surfaceConn.size(); iNode++) surfaceConn[iNode] = wallVertices[surfaceConn[iNode]];
      
    }

    CADTElemClass ElemADT(nDim, surfaceCoor, surfaceConn, VTK_TypeElem, markerIDs, elemIDs, true);

    /*--- Looping over all nodes and check if part of domain and not on boundary ---*/
    unsigned short nearest_marker; unsigned long nearest_elem;
    for (auto iNode = 0ul; iNode < geometry->GetnPoint(); iNode++) {    
      if (!geometry->nodes->GetBoundary(iNode)) {
        ElemADT.DetermineNearestElement(geometry->nodes->GetCoord(iNode), dist, nearest_marker, nearest_elem, rankID);
        // cout << geometry->nodes->GetCoord(iNode)[0] << "\t" << geometry->nodes->GetCoord(iNode)[1] << "\t"<<  dist << "\t" << nearest_marker << "\t" << nearest_elem << endl;
        if(abs(dist-IL_height) < 1e-12){
          // cout << geometry->nodes->GetCoord(iNode)[0] << "\t" << geometry->nodes->GetCoord(iNode)[1] << "\t"<<  dist << "\t" << nearest_marker << "\t" << nearest_elem << endl;
          IL_EdgeNodes.push_back(new CRadialBasisFunctionNode(iNode, nearest_marker, nearest_elem));
        }else if(dist-IL_height < 0){
          IL_internalNodes[nearest_marker]->push_back(iNode);
        }else{
          internalNodes.push_back(iNode);
        }
      }   
    }  

    //TODO remove these checks
    ofstream ofile;
    ofile.open("il_wall_nodes.txt");
    for(auto x : IL_WallNodes){
      ofile << x->GetIndex() << endl;
    }
    ofile.close();

    ofile.open("il_edge_nodes.txt");
    for(auto x : IL_EdgeNodes){
      ofile << x->GetIndex() << endl;
    }
    ofile.close();

    ofile.open("il_internal_nodes.txt");
    // for(auto x : IL_internalNodes){
    for(auto y : *IL_internalNodes[0]){
      ofile << y << endl;
    }
    // }
    ofile.close();
  }else{

    /*--- Looping over all nodes and check if part of domain and not on boundary ---*/
    for (auto iNode = 0ul; iNode < geometry->GetnPoint(); iNode++) {    
      if (!geometry->nodes->GetBoundary(iNode)) {
        internalNodes.push_back(iNode);
      }   
    }  
  }

  /*--- Adding nodes on markers considered as internal nodes ---*/
  for (auto iMarker = 0u; iMarker < geometry->GetnMarker(); iMarker++){

    /*--- Check if marker is considered as internal nodes ---*/
    if(config->GetMarker_All_Deform_Mesh_Internal(iMarker)){
      
      /*--- Loop over marker vertices ---*/
      for (auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++) { 

        /*--- Local node index ---*/
        auto iNode = geometry->vertex[iMarker][iVertex]->GetNode();

        /*--- if not among the boundary nodes ---*/
        if (find_if (BoundNodes.begin(), BoundNodes.end(), [&](CRadialBasisFunctionNode* i){return i->GetIndex() == iNode;}) == BoundNodes.end()) {
          internalNodes.push_back(iNode);
        }            
      }
    }
  }

  /*--- In case of a parallel computation, the nodes on the send/receive markers are included as internal nodes
          if they are not already a boundary node with known deformation ---*/

  #ifdef HAVE_MPI
    /*--- Looping over the markers ---*/
    for (auto iMarker = 0u; iMarker < geometry->GetnMarker(); iMarker++) { 

      /*--- If send or receive marker ---*/
      if (config->GetMarker_All_SendRecv(iMarker)) { 

        /*--- Loop over marker vertices ---*/
        for (auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++) { 
          
          /*--- Local node index ---*/
          auto iNode = geometry->vertex[iMarker][iVertex]->GetNode();

          /*--- if not among the boundary nodes ---*/
          if (find_if (BoundNodes.begin(), BoundNodes.end(), [&](CRadialBasisFunctionNode* i){return i->GetIndex() == iNode;}) == BoundNodes.end()) {
            internalNodes.push_back(iNode);
          }             
        }
      }
    }

    /*--- sorting of the local indices ---*/
    sort(internalNodes.begin(), internalNodes.end());

    /*--- Obtaining unique set of internal nodes ---*/
    internalNodes.resize(std::distance(internalNodes.begin(), unique(internalNodes.begin(), internalNodes.end())));
  #endif
}


void CRadialBasisFunctionInterpolation::ComputeInterpCoeffs(su2passivematrix& invInterpMat){

  /*--- resizing the interpolation coefficient vector ---*/
  InterpCoeff.resize(nDim*nCtrlNodesGlobal);

  /*--- Coefficients are found on the master process.
          Resulting coefficient is found by summing the multiplications of inverse interpolation matrix entries with deformation ---*/
  if(rank == MASTER_NODE){    

    for(auto iNode = 0ul; iNode < nCtrlNodesGlobal; iNode++){
      for (auto iDim = 0u; iDim < nDim; iDim++){
        InterpCoeff[iNode*nDim + iDim] = 0;
        for (auto jNode = 0ul; jNode < nCtrlNodesGlobal; jNode++){        
          InterpCoeff[iNode * nDim + iDim] += invInterpMat(iNode,jNode) * CtrlNodeDeformation[jNode*nDim+ iDim]; 
        }
      }
    }
  }
  
  /*--- Broadcasting the interpolation coefficients ---*/
  #ifdef HAVE_MPI
    SU2_MPI::Bcast(InterpCoeff.data(), InterpCoeff.size(), MPI_DOUBLE, MASTER_NODE, SU2_MPI::GetComm());
  #endif  
}

void CRadialBasisFunctionInterpolation::UpdateGridCoord(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, const vector<unsigned long>& internalNodes){
  
  if(rank == MASTER_NODE){
    cout << "updating the grid coordinates" << endl;
  }

  /*--- Update of internal node coordinates ---*/
  UpdateInternalCoords(geometry, type, radius, internalNodes);

  /*--- Update of boundary node coordinates ---*/
  UpdateBoundCoords(geometry, config, type, radius);   

  /*--- In case of data reduction, perform the correction for nonzero error nodes ---*/
  if(config->GetRBF_DataReduction() && BoundNodes.size() > 0){
    SetCorrection(geometry, config, type, internalNodes);
  }
}

void CRadialBasisFunctionInterpolation::UpdateInternalCoords(CGeometry* geometry, const RADIAL_BASIS& type, const su2double radius, const vector<unsigned long>& internalNodes){
  
   /*--- Vector for storing the coordinate variation ---*/
  su2double var_coord[nDim]{0.0};
  
  /*--- Loop over the internal nodes ---*/
  for(auto iNode = 0ul; iNode < internalNodes.size(); iNode++){

    /*--- Loop for contribution of each control node ---*/
    for(auto jNode = 0ul; jNode < nCtrlNodesGlobal; jNode++){

      /*--- Determine distance between considered internal and control node ---*/
      auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(internalNodes[iNode]));

      /*--- Evaluate RBF based on distance ---*/
      auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));
      
      /*--- Add contribution to total coordinate variation ---*/
      for(auto iDim = 0u; iDim < nDim; iDim++){
        var_coord[iDim] += rbf*InterpCoeff[jNode * nDim + iDim];
      }
    }

    /*--- Apply the coordinate variation and resetting the var_coord vector to zero ---*/
    for(auto iDim = 0u; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(internalNodes[iNode], iDim, var_coord[iDim]);
      var_coord[iDim] = 0;
    } 
  }  
}

void CRadialBasisFunctionInterpolation::UpdateBoundCoords(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius){
  
  /*--- Vector for storing the coordinate variation ---*/
  su2double var_coord[nDim]{0.0};
  
  /*--- In case of data reduction, the non-control boundary nodes are treated as if they where internal nodes ---*/
  if(config->GetRBF_DataReduction()){

    /*--- Looping over the non selected boundary nodes ---*/
    for(auto iNode = 0ul; iNode < BoundNodes.size(); iNode++){
      
      /*--- Finding contribution of each control node ---*/
      for( auto jNode = 0ul; jNode <  nCtrlNodesGlobal; jNode++){
        
        /*--- Distance of non-selected boundary node to control node ---*/
        auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(BoundNodes[iNode]->GetIndex()));
        
        /*--- Evaluation of the radial basis function based on the distance ---*/
        auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));

        /*--- Computing and add the resulting coordinate variation ---*/
        for(auto iDim = 0u; iDim < nDim; iDim++){
          var_coord[iDim] += rbf*InterpCoeff[jNode * nDim + iDim];
        }
      }

      /*--- Applying the coordinate variation and resetting the var_coord vector*/
      for(auto iDim = 0u; iDim < nDim; iDim++){
        geometry->nodes->AddCoord(BoundNodes[iNode]->GetIndex(), iDim, var_coord[iDim]);
        var_coord[iDim] = 0;
      }
    }
  }
  
  /*--- Applying the surface deformation, which are stored in the deformation vector ---*/
  unsigned long idx = 0;
  for ( auto iControlNodes : ControlNodes){
    for(auto jNode = 0ul; jNode < iControlNodes->size(); jNode++){ 
      for(auto iDim = 0u; iDim < nDim; iDim++){
          geometry->nodes->AddCoord((*iControlNodes)[jNode]->GetIndex(), iDim, CtrlNodeDeformation[idx++]); 
      }
    }
  } 
}


void CRadialBasisFunctionInterpolation::GetInitMaxErrorNode(CGeometry* geometry, CConfig* config, vector<unsigned long>& maxErrorNodes, su2double& maxErrorLocal){

  /*--- Set max error to zero ---*/
  maxErrorLocal = 0.0;

  unsigned long maxErrorNodeLocal;
  /*--- Loop over the nodes ---*/  
  for(auto iNode = 0ul; iNode < BoundNodes.size(); iNode++){

    /*--- Compute to squared norm of the deformation ---*/
    su2double normSquaredDeformation = GeometryToolbox::SquaredNorm(nDim, geometry->vertex[BoundNodes[iNode]->GetMarker()][BoundNodes[iNode]->GetVertex()]->GetVarCoord());    

    /*--- In case squared norm deformation is larger than the error, update the error ---*/
    if(normSquaredDeformation > maxErrorLocal){
      maxErrorLocal = normSquaredDeformation;
      maxErrorNodeLocal = iNode;
    }
  }
  
  if(maxErrorLocal > 0){
    maxErrorNodes.push_back(maxErrorNodeLocal);

    /*--- Account for the possibility of applying the deformation in multiple steps ---*/
    maxErrorLocal = sqrt(maxErrorLocal) / ((su2double)config->GetGridDef_Nonlinear_Iter());


    /*--- Steps for finding a node on a secondary edge (double edged greedy algorithm) ---*/

    /*--- Total error, defined as total variation in coordinates ---*/
    su2double* errorTotal = geometry->vertex[BoundNodes[maxErrorNodeLocal]->GetMarker()][BoundNodes[maxErrorNodeLocal]->GetVertex()]->GetVarCoord();

    /*--- Making a copy of the total error to errorStep, to allow for manipulation of the data ---*/
    su2double* errorStep = new su2double[nDim];
    std::copy(errorTotal, errorTotal+nDim, errorStep);

    /*--- Account for applying deformation in multiple steps ---*/
    for( unsigned short iDim = 0u; iDim < nDim; iDim++){
      errorStep[iDim] = errorStep[iDim]/(su2double)config->GetGridDef_Nonlinear_Iter();
    }
    
    /*--- Finding a double edged error node if possible ---*/
    GetDoubleEdgeNode(errorStep, maxErrorNodes);
  }
}


void CRadialBasisFunctionInterpolation::SetCtrlNodeCoords(CGeometry* geometry){
  /*--- The coordinates of all control nodes are made available on all processes ---*/
  
  /*--- resizing the matrix containing the global control node coordinates ---*/
  CtrlCoords.resize(nCtrlNodesGlobal*nDim);
  
  /*--- Array containing the local control node coordinates ---*/ 
  su2double localCoords[nDim * nCtrlNodesLocal];
  
  /*--- Storing local control node coordinates ---*/
  unsigned long idx = 0;
  for(auto iControlNodes : ControlNodes){
    for( auto iNode = 0ul; iNode < iControlNodes->size(); iNode++ ){
      auto coord = geometry->nodes->GetCoord((*iControlNodes)[iNode]->GetIndex());  
      for ( auto iDim = 0u ; iDim < nDim; iDim++ ){
        localCoords[ idx++ ] = coord[iDim];
      }
    }
  }

  /*--- Gathering local control node coordinate sizes on all processes. ---*/
  int LocalCoordsSizes[size];
  int localCoordsSize = nDim * nCtrlNodesLocal;
  SU2_MPI::Allgather(&localCoordsSize, 1, MPI_INT, LocalCoordsSizes, 1, MPI_INT, SU2_MPI::GetComm()); 

  /*--- Array containing the starting indices for the allgatherv operation */
  int disps[SU2_MPI::GetSize()] = {0};    

  for(auto iProc = 1; iProc < SU2_MPI::GetSize(); iProc++){
    disps[iProc] = disps[iProc-1]+LocalCoordsSizes[iProc-1];
  }
  
  /*--- Distributing global control node coordinates among all processes ---*/
  SU2_MPI::Allgatherv(&localCoords, localCoordsSize, MPI_DOUBLE, CtrlCoords.data(), LocalCoordsSizes, disps, MPI_DOUBLE, SU2_MPI::GetComm()); 
};


void CRadialBasisFunctionInterpolation::GetInterpError(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, su2double& maxErrorLocal, vector<unsigned long>& maxErrorNodes){
  
  /*--- Array containing the local error ---*/
  su2double localError[nDim];

  /*--- Magnitude of the local maximum error ---*/
  maxErrorLocal = 0.0;
  unsigned long maxErrorNodeLocal;
  /*--- Loop over non-selected boundary nodes ---*/
  for(auto iNode = 0ul; iNode < BoundNodes.size(); iNode++){

    /*--- Compute nodal error ---*/
    GetNodalError(geometry, config, type, radius, iNode, localError);

    /*--- Setting error ---*/
    BoundNodes[iNode]->SetError(localError, nDim);

    /*--- Compute error magnitude and update local maximum error if necessary ---*/
    su2double errorMagnitude = GeometryToolbox::Norm(nDim, localError);
    if(errorMagnitude > maxErrorLocal){
      maxErrorLocal = errorMagnitude;
      maxErrorNodeLocal = iNode;
    }
  }  

  if(maxErrorLocal > 0){
    /*--- Including the maximum error nodes in the max error nodes vector ---*/
    maxErrorNodes.push_back(maxErrorNodeLocal);
    
    /*--- Finding a double edged error node if possible ---*/
    GetDoubleEdgeNode(BoundNodes[maxErrorNodes[0]]->GetError(), maxErrorNodes);
  }
}

void CRadialBasisFunctionInterpolation::GetNodalError(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, unsigned long iNode, su2double* localError){ 
  
  /*--- If requested (no by default) impose the surface deflections in increments ---*/
  const su2double VarIncrement = 1.0 / ((su2double)config->GetGridDef_Nonlinear_Iter());
  
  /*--- If node is part of a moving boundary then the error is defined as the difference
           between the found and prescribed displacements. Thus, here the displacement is substracted from the error ---*/
  if(config->GetMarker_All_Moving(BoundNodes[iNode]->GetMarker())){
    auto displacement = geometry->vertex[BoundNodes[iNode]->GetMarker()][BoundNodes[iNode]->GetVertex()]->GetVarCoord();

    for(auto iDim = 0u; iDim < nDim; iDim++){
      localError[iDim] = -displacement[iDim] * VarIncrement;
    }
  }else{
    for(auto iDim = 0u; iDim < nDim; iDim++){
      localError[iDim] = 0; 
    }
  }

  /*--- Resulting displacement from the RBF interpolation is added to the error ---*/

  /*--- Finding contribution of each control node ---*/
  for(auto jNode = 0ul; jNode < nCtrlNodesGlobal; jNode++){

    /*--- Distance between non-selected boundary node and control node ---*/
    auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode *nDim], geometry->nodes->GetCoord(BoundNodes[iNode]->GetIndex()));

    /*--- Evaluation of Radial Basis Function ---*/
    auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));

    /*--- Add contribution to error ---*/
    for(auto iDim = 0u; iDim < nDim; iDim++){
      localError[iDim] += rbf*InterpCoeff[jNode*nDim + iDim];
    }
  }

}

void CRadialBasisFunctionInterpolation::GetDoubleEdgeNode(const su2double* maxError, vector<unsigned long>& maxErrorNodes){
  
  /*--- Obtaining maximum error vector and its corresponding angle ---*/
  const auto polarAngleMaxError = atan2(maxError[1],maxError[0]);
  const auto azimuthAngleMaxError = atan2( sqrt( pow(maxError[0],2) + pow(maxError[1],2)), maxError[2]);

  su2double max = 0;
  unsigned long idx;
  bool found = false;

  for(auto iNode = 0ul; iNode < BoundNodes.size(); iNode++){
    
    auto error = BoundNodes[iNode]->GetError();
    su2double polarAngle = atan2(error[1],error[0]);
    su2double relativePolarAngle = abs(polarAngle - polarAngleMaxError);

    switch(nDim){
      case 2:
        if ( abs(relativePolarAngle - M_PI) < M_PI/2 ){
          CompareError(error, iNode, max, idx);
        }
        break;
      case 3:

        su2double azimuthAngle = atan2( sqrt( pow(error[0],2) + pow(error[1],2)), error[2]);
        if ( abs(relativePolarAngle - M_PI) < M_PI/2 ){
          if( azimuthAngleMaxError <= M_PI/2 ) {
            if ( azimuthAngle > M_PI/2 - azimuthAngleMaxError) {
              CompareError(error, iNode, max, idx);
            }
          }
          else{
            if ( azimuthAngle < 1.5 * M_PI - azimuthAngleMaxError ){
              CompareError(error, iNode, max, idx);
            }
          }          
        }
        else{
          if(azimuthAngleMaxError <= M_PI/2){
            if(azimuthAngle > M_PI/2 + azimuthAngleMaxError){
              CompareError(error, iNode, max, idx);
            }
          }else{
            if(azimuthAngle < azimuthAngleMaxError -  M_PI/2){
              CompareError(error, iNode, max, idx);
            }
          }
        }
        break;
    }
  }
  

  /*--- Include the found double edge node in the maximum error nodes vector ---*/
  if(max > 0){
    maxErrorNodes.push_back(idx);
  }
}

void CRadialBasisFunctionInterpolation::CompareError(su2double* error, unsigned long iNode, su2double& maxError, unsigned long& idx){
  auto errMag = GeometryToolbox::SquaredNorm(nDim, error);

  if(errMag > maxError){
    maxError = errMag;
    idx = iNode;
  }
}

void CRadialBasisFunctionInterpolation::SetCorrection(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const vector<unsigned long>& internalNodes){

  /*--- The non-selected control nodes still have a nonzero error once the maximum error falls below the data reduction tolerance. 
          This error is applied as correction and interpolated into the volumetric mesh for internal nodes that fall within the correction radius.
          To evaluate whether an internal node falls within the correction radius an AD tree is constructed of the boundary nodes,
          making it possible to determine the distance to the nearest boundary node. ---*/

  /*--- Construction of the AD tree consisting of the non-selected boundary nodes ---*/

  /*--- Number of non-selected boundary nodes ---*/
  const unsigned long nVertexBound = BoundNodes.size();
  
  /*--- Vector storing the coordinates of the boundary nodes ---*/
  vector<su2double> Coord_bound(nDim*nVertexBound);

  /*--- Vector storing the IDs of the boundary nodes ---*/
  vector<unsigned long> PointIDs(nVertexBound);

  /*--- Correction Radius, equal to maximum error times a prescribed constant ---*/
  const su2double CorrectionRadius = config->GetRBF_DataRedCorrectionFactor()*MaxErrorGlobal; 

  /*--- Storing boundary node information ---*/
  unsigned long i = 0;
  unsigned long j = 0;
  for(auto iVertex = 0ul; iVertex < nVertexBound; iVertex++){
    auto iNode = BoundNodes[iVertex]->GetIndex();
    PointIDs[i++] = iVertex;
    for(auto iDim = 0u; iDim < nDim; iDim++){
      Coord_bound[j++] = geometry->nodes->GetCoord(iNode, iDim);
    }
  }

  /*--- Construction of AD tree ---*/
  CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);

  /*--- ID of nearest boundary node ---*/
  unsigned long pointID;
  /*--- Distance to nearest boundary node ---*/
  su2double dist;
  /*--- rank of nearest boundary node ---*/
  int rankID;
 
  /*--- Interpolation of the correction to the internal nodes that fall within the correction radius ---*/
  for(auto iNode = 0ul; iNode < internalNodes.size(); iNode++){

    /*--- Find nearest node ---*/
    BoundADT.DetermineNearestNode(geometry->nodes->GetCoord(internalNodes[iNode]), dist, pointID, rankID);  
    
    /*--- Get error of nearest node ---*/
    auto err = BoundNodes[pointID]->GetError();

    /*--- evaluate RBF ---*/    
    auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, CorrectionRadius, dist));

    /*--- Apply correction to the internal node ---*/
    for(auto iDim = 0u; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(internalNodes[iNode], iDim, -rbf*err[iDim]);
    }
  }
  


  /*--- Applying the correction to the non-selected boundary nodes ---*/
  for(auto iNode = 0ul; iNode < BoundNodes.size(); iNode++){
    auto err =  BoundNodes[iNode]->GetError();
    for(auto iDim = 0u; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(BoundNodes[iNode]->GetIndex(), iDim, -err[iDim]);
    }
  }
}


void CRadialBasisFunctionInterpolation::AddControlNode(vector<unsigned long>& maxErrorNodes){
  /*--- Sort indices in descending order, to prevent shift in index as they are erased ---*/
  sort(maxErrorNodes.rbegin(),maxErrorNodes.rend()); 
  
  for(auto iNode : maxErrorNodes){
    /*--- Addition of node to the reduced set of control nodes ---*/
    ReducedControlNodes.push_back(move(BoundNodes[iNode]));

    /*--- Removal of node among the non-selected boundary nodes ---*/
    BoundNodes.erase(BoundNodes.begin()+iNode);
  }

  /*--- Clearing maxErrorNodes vector ---*/
  maxErrorNodes.clear();
 
}


void CRadialBasisFunctionInterpolation::Get_nCtrlNodes(){
  /*--- Determining the global number of control nodes ---*/
  
  /*--- Local number of control nodes ---*/
  nCtrlNodesLocal = 0;
  for(auto iControlNodes : ControlNodes) { nCtrlNodesLocal += iControlNodes->size(); }
  // unsigned long local_nControlNodes = ControlNodes->size();

  /*--- Summation of local number of control nodes ---*/
  SU2_MPI::Allreduce(&nCtrlNodesLocal, &nCtrlNodesGlobal, 1, MPI_UNSIGNED_LONG, MPI_SUM, SU2_MPI::GetComm());

}

void CRadialBasisFunctionInterpolation::GetIL_Deformation(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius){
 

  /*--- Solve for interpolation coefficient using solely wall nodes as ctrl nodes ---*/
  GetInterpCoeffs(geometry, config, type, radius);
  
  /*--- Obtain the required deformation of the inflation layer edge ---*/
  GetIL_EdgeDeformation(geometry, config, type, radius);

  /*--- Adding the inflation layer edge nodes to the control nodes ---*/
  ControlNodes.push_back(&IL_EdgeNodes);
  Get_nCtrlNodes();

  /*--- Solve for interpolation coefficients ---*/
  GetInterpCoeffs(geometry, config, type, radius);

  /*--- Updating the inflation layer coordinates ---*/ 
  UpdateInflationLayerCoords(geometry, type, radius);
  
  /*--- Set control nodes for the domain outside inflation layer ---*/
  ControlNodes.resize(2);
  ControlNodes[0] = &BoundNodes;
  ControlNodes[1] = &IL_EdgeNodes;  
  Get_nCtrlNodes();
}

void CRadialBasisFunctionInterpolation::GetIL_EdgeDeformation(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius){

  /*--- Inflation layer height ---*/
  su2double IL_height = config->GetRBF_IL_Height();

  /*--- Number of deformation steps ---*/
  auto Nonlinear_iter = config->GetGridDef_Nonlinear_Iter();

  /*--- Obtaining free displacement of the inflation layer edge nodes ---*/
  for(auto iNode = 0ul; iNode < IL_EdgeNodes.size(); iNode++){
    
    /*--- Obtaining coordinates ---*/
    auto coord = geometry->nodes->GetCoord(IL_EdgeNodes[iNode]->GetIndex());

    /*--- Setting new coord equal to old coord ---*/
    IL_EdgeNodes[iNode]->SetNewCoord(coord, nDim);

    /*--- Loop for contribution of each control node ---*/
    for(auto jNode = 0ul; jNode < nCtrlNodesGlobal; jNode++){

      /*--- Determine distance between considered internal and control node ---*/
      auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(IL_EdgeNodes[iNode]->GetIndex()));

      /*--- Evaluate RBF based on distance ---*/
      auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));
      
      /*--- Add contribution to new coordinates -- -*/
      for(auto iDim = 0u; iDim < nDim; iDim++){
        IL_EdgeNodes[iNode]->AddNewCoord(rbf*InterpCoeff[jNode*nDim+iDim], iDim);
      }
    }
  }  

  /*--- Assembly of AD tree containing the updated wall nodes ---*/
  vector<su2double> Coord_bound(nDim * IL_WallNodes.size());
  vector<unsigned long> PointIDs(IL_WallNodes.size());
  unsigned long pointID;
  su2double dist; 
  int rankID;
  unsigned long ii = 0;

  /*--- Loop through wall nodes ---*/
  for( auto jNode = 0ul; jNode < IL_WallNodes.size(); jNode++){

    /*--- Assign identifier ---*/
    PointIDs[jNode] = jNode;

    for(auto iDim = 0u; iDim < nDim; iDim++){

      /*--- Applying the deformation ---*/
      geometry->nodes->AddCoord(IL_WallNodes[jNode]->GetIndex(), iDim, CtrlNodeDeformation[jNode * nDim + iDim]);
      
      /*--- store updated position ---*/
      Coord_bound[ii++] = geometry->nodes->GetCoord(IL_WallNodes[jNode]->GetIndex())[iDim];
    }
  }

  /*--- Update of boundary to obtain normals of updated geometry ---*/
  geometry->SetBoundControlVolume(config, UPDATE);

  /*--- AD tree with updated wall positions ---*/
  CADTPointsOnlyClass WallADT(nDim, IL_WallNodes.size(), Coord_bound.data(), PointIDs.data(), true);
  
  /*--- Finding the required displacement of the edge nodes ---*/

  /*--- Distance to nearest wall node and required added inflation layer thickness ---*/
  su2double dist_vec[nDim];
  su2double added_thickness;

  /*--- Loop over inflation layer wall nodes ---*/
  for(auto iNode = 0ul; iNode < IL_EdgeNodes.size(); iNode++){      
    
    /*--- Get nearest wall node ---*/
    WallADT.DetermineNearestNode(IL_EdgeNodes[iNode]->GetNewCoord(), dist, pointID, rankID);

    /*--- Get normal and make it a unit vector ---*/
    auto normal = geometry->vertex[IL_WallNodes[pointID]->GetMarker()][IL_WallNodes[pointID]->GetVertex()]->GetNormal(); 
    auto normal_length = GeometryToolbox::Norm(nDim, normal);
    for(auto iDim = 0u; iDim < nDim; iDim++){
      normal[iDim] = normal[iDim]/normal_length;
    }

    /*--- Get distance vector from edge node to nearest wall node ---*/
    GeometryToolbox::Distance(nDim, IL_EdgeNodes[iNode]->GetNewCoord(), geometry->nodes->GetCoord(IL_WallNodes[pointID]->GetIndex()), dist_vec);

    /*--- Dot product to obtain current inflation layer height ---*/
    auto dp = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

    /*--- Get required change in inflation layer thickness ---*/
    added_thickness = - IL_height + abs(dp); // TODO sign keeps changing somehow (started as +, - for 3D | -, + for 2D)

    /*--- Apply required change in coordinates and store variation w.r.t. initial coordinates. ---*/
    su2double var_coord[nDim];
    for(auto iDim = 0u; iDim < nDim; iDim++){
      IL_EdgeNodes[iNode]->AddNewCoord(added_thickness * normal[iDim], iDim);
      var_coord[iDim] = (IL_EdgeNodes[iNode]->GetNewCoord()[iDim] - geometry->nodes->GetCoord(IL_EdgeNodes[iNode]->GetIndex())[iDim])*Nonlinear_iter;
    }

    IL_EdgeNodes[iNode]->SetVarCoord(var_coord, nDim);
  }

  /*--- Set wall back to initial position for accurate calculation of RBFs ---*/
  for( auto jNode = 0ul; jNode < IL_WallNodes.size(); jNode++){
    for(auto iDim = 0u; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(IL_WallNodes[jNode]->GetIndex(), iDim, -CtrlNodeDeformation[jNode * nDim + iDim]);
    }
  } 

  /*--- Update wall boundary ---*/
  geometry->SetBoundControlVolume(config, UPDATE);
}

void CRadialBasisFunctionInterpolation::UpdateInflationLayerCoords(CGeometry* geometry, const RADIAL_BASIS& type, const su2double radius){
  
  /*--- Vector for storing the coordinate variation ---*/
  su2double var_coord[nDim]{0.0};
  
  /*--- Loop over the internal nodes ---*/
  for(auto iNode = 0ul; iNode < IL_internalNodes[0]->size(); iNode++){

    /*--- Loop for contribution of each control node ---*/
    for(auto jNode = 0ul; jNode < nCtrlNodesGlobal; jNode++){

      /*--- Determine distance between considered internal and control node ---*/
      auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord((*IL_internalNodes[0])[iNode]));

      /*--- Evaluate RBF based on distance ---*/
      auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));
      
      /*--- Add contribution to total coordinate variation ---*/
      for(auto iDim = 0u; iDim < nDim; iDim++){
        var_coord[iDim] += rbf*InterpCoeff[jNode * nDim + iDim];
      }
    }

    /*--- Apply the coordinate variation and resetting the var_coord vector to zero ---*/
    for(auto iDim = 0u; iDim < nDim; iDim++){
      geometry->nodes->AddCoord((*IL_internalNodes[0])[iNode], iDim, var_coord[iDim]);
      var_coord[iDim] = 0;
    } 
  }  
  
  /*--- Applying the wall deformation ---*/
  for(auto jNode = 0ul; jNode < (*ControlNodes[0]).size(); jNode++){ 
    for(auto iDim = 0u; iDim < nDim; iDim++){
        geometry->nodes->AddCoord((*ControlNodes[0])[jNode]->GetIndex(), iDim, CtrlNodeDeformation[jNode * nDim + iDim]); 
    }
  }

}