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
  
  // if(rank == 0){
  //   cout << "internal nodes: " << endl;
  //   for (auto x : internalNodes){
  //     cout << geometry->nodes->GetGlobalIndex(x) << endl;
  //   }
  // }

  
  /*--- Looping over the number of deformation iterations ---*/
  for (auto iNonlinear_Iter = 0ul; iNonlinear_Iter <  Nonlinear_Iter; iNonlinear_Iter++) {
    
    SetCtrlNodes(config); // placed here for the sliding algo

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
    vector <CRadialBasisFunctionNode*> maxErrorNodes;

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
    while(MaxErrorGlobal > DataReductionTolerance || greedyIter == 0 ){ 
      
      /*--- In case of a nonzero local error, control nodes are added ---*/
      if(maxErrorLocal> 0){
        AddControlNode(maxErrorNodes);
      }

      /*--- Obtaining the global number of control nodes. ---*/
      Get_nCtrlNodesGlobal();

      if(ReducedSlideNodes.size() > 0){

        /*--- Vector storing the coordinates of the boundary nodes ---*/
        vector<su2double> Coord_bound;

        /*--- Vector storing the IDs of the boundary nodes ---*/
        vector<unsigned long> PointIDs;

        // for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++) {
        //   if(config->GetMarker_All_Deform_Mesh_Slide(iMarker)){
        for ( auto iVertex = 0ul; iVertex < SlideSurfNodes.size(); iVertex++){

          // auto iNode = geometry->vertex[iMarker][iVertex]->GetNode();
          auto iNode = SlideSurfNodes[iVertex]->GetIndex();
          PointIDs.push_back(iVertex); //TODO
          for(auto iDim = 0u; iDim < nDim; iDim++){
            Coord_bound.push_back(geometry->nodes->GetCoord(iNode, iDim));
          }
          // cout << iNode << " " << Coord_bound[Coord_bound.size()-3] << " " << Coord_bound[Coord_bound.size()-2] << " " << Coord_bound[Coord_bound.size()-1] << endl;
        }

        for (auto iVertex = 0ul; iVertex < ReducedSlideNodes.size(); iVertex++){
          auto iNode = ReducedSlideNodes[iVertex]->GetIndex();
          PointIDs.push_back(iVertex+SlideSurfNodes.size()); //TODO
          for(auto iDim = 0u; iDim < nDim; iDim++){
            Coord_bound.push_back(geometry->nodes->GetCoord(iNode, iDim));
          }
        }
        
        //   }
        // }

        /*--- Number of non-selected boundary nodes ---*/
        const unsigned long nVertexBound = PointIDs.size();

        /*--- Construction of AD tree ---*/
        CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);

        ControlNodes.resize(1);
        Get_nCtrlNodesGlobal();
        GetSlidingDeformation(geometry, config, type, radius, BoundADT);
        ControlNodes.push_back(&ReducedSlideNodes);
        Get_nCtrlNodesGlobal();
      }

      /*--- Obtaining the interpolation coefficients. ---*/
      GetInterpCoeffs(geometry, config, type, radius);
      
      /*--- Determining the interpolation error, of the non-control boundary nodes. ---*/
      GetInterpError(geometry, config, type, radius, maxErrorLocal, maxErrorNodes);       
      SU2_MPI::Allreduce(&maxErrorLocal, &MaxErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());

      if(rank == MASTER_NODE) cout << "Greedy iteration: " << greedyIter << ". Max error: " << MaxErrorGlobal << ". Tol: " << DataReductionTolerance << ". Global nr. of ctrl nodes: "  << nCtrlNodesGlobal << "\n" << endl;
      greedyIter++;

    }  
  }else{
    
    if(SlideSurfNodes.size() > 0){
      
      /*--- Vector storing the coordinates of the boundary nodes ---*/
      vector<su2double> Coord_bound;

      /*--- Vector storing the IDs of the boundary nodes ---*/
      vector<unsigned long> PointIDs;

      // for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++) {
      //   if(config->GetMarker_All_Deform_Mesh_Slide(iMarker)){
      for ( auto iVertex = 0ul; iVertex < SlideSurfNodes.size(); iVertex++){

        // auto iNode = geometry->vertex[iMarker][iVertex]->GetNode();
        auto iNode = SlideSurfNodes[iVertex]->GetIndex();
        PointIDs.push_back(iVertex); //TODO
        for(auto iDim = 0u; iDim < nDim; iDim++){
          Coord_bound.push_back(geometry->nodes->GetCoord(iNode, iDim));
        }
        // cout << iNode << " " << Coord_bound[Coord_bound.size()-3] << " " << Coord_bound[Coord_bound.size()-2] << " " << Coord_bound[Coord_bound.size()-1] << endl;
      }
      
      //   }
      // }

      /*--- Number of non-selected boundary nodes ---*/
      const unsigned long nVertexBound = PointIDs.size();

      /*--- Construction of AD tree ---*/
      CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);

      GetSlidingDeformation(geometry, config, type, radius, BoundADT);
      ControlNodes.push_back(&SlideSurfNodes);
      Get_nCtrlNodesGlobal();
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
    if (!config->GetMarker_All_Deform_Mesh_Internal(iMarker) && !config->GetMarker_All_SendRecv(iMarker) && !config->GetMarker_All_Deform_Mesh_Slide(iMarker)){

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
  }

  //loop for sliding nodes

  // loop over markers
  for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++) {

    // if sliding marker
    if(config->GetMarker_All_Deform_Mesh_Slide(iMarker)){

      //loop over vertices
      for ( auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++){

        // node in questin
        auto iNode = geometry->vertex[iMarker][iVertex]->GetNode();
        
        // count the number of markers it is involved in
        unsigned short nVertex = 0;
        for(unsigned short i = 0; i < config->GetnMarker_All(); i++){ //TODO change to jMarker
          if(geometry->nodes->GetVertex(geometry->vertex[iMarker][iVertex]->GetNode(), i) != -1){
            if(!config->GetMarker_All_SendRecv(i)){
              nVertex++;
            }
            
          } 
        }

        if(nVertex > 1){ //TODO this should only be allowed if not a send receive marker
          if(geometry->nodes->GetDomain(iNode)){
            BoundNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
          }
        }else{
          if(find_if (BoundNodes.begin(), BoundNodes.end(), [&](CRadialBasisFunctionNode* i){return i->GetIndex() == iNode;}) == BoundNodes.end()){
            if (geometry->nodes->GetDomain(iNode)) {
              SlideSurfNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
            } 
            
          }
        }
      }
    }
  }

  /*--- Sorting of the boundary nodes based on their index ---*/
  stable_sort(BoundNodes.begin(), BoundNodes.end(), HasSmallerIndex);
  sort(SlideSurfNodes.begin(), SlideSurfNodes.end(), HasSmallerIndex);

  /*--- Obtaining unique set ---*/
  
  BoundNodes.resize(std::distance(BoundNodes.begin(), unique(BoundNodes.begin(), BoundNodes.end(), HasEqualIndex)));

  // if(rank == 0){
  //   cout << "sliding surf nodes: " << endl;
  //   for(auto x : SlideSurfNodes){
  //     cout << geometry->nodes->GetGlobalIndex(x->GetIndex()) << endl;
  //   }

  //   cout << "bound nodes: " << endl;
  //   for(auto x : BoundNodes){
  //     cout << geometry->nodes->GetGlobalIndex(x->GetIndex()) << endl;
  //   }
  // }
  ofstream myfile;
  myfile.open ("BoundNodes.txt");
  for(auto x  : BoundNodes){ myfile << x->GetIndex() << "\t" << endl;}
  myfile.close();

  ofstream myfile2;
  myfile2.open ("SlideNodes.txt");
  for(auto x  : SlideSurfNodes){ myfile2 << x->GetIndex() << "\t" << endl;}
  myfile2.close();

  cout << "nr of bound nodes: " << BoundNodes.size() << endl;
  cout << "nr of slide nodes: " << SlideSurfNodes.size() << endl;
  
}

void CRadialBasisFunctionInterpolation::SetCtrlNodes(CConfig* config){
  
  /*--- Assigning the control nodes based on whether data reduction is applied or not. ---*/
  if(config->GetRBF_DataReduction()){
    
    ControlNodes.resize(1); //TODO can likely be done more elegantly
    /*--- Control nodes are an empty set ---*/
    ControlNodes[0] = &ReducedBoundNodes;
    if(SlideSurfNodes.size() > 0){
      ControlNodes.push_back(&ReducedSlideNodes);
    }
  }else{
    ControlNodes.resize(1);
    /*--- Control nodes are the boundary nodes ---*/
    ControlNodes[0] = &BoundNodes;
  }

  /*--- Obtaining the total number of control nodes. ---*/
  Get_nCtrlNodesGlobal();

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
  CtrlNodeDeformation.resize(nCtrlNodesLocal*nDim, 0.0); 

  /*--- If requested (no by default) impose the surface deflections in
    increments and solve the grid deformation with
    successive small deformations. ---*/
  const su2double VarIncrement = 1.0 / ((su2double)config->GetGridDef_Nonlinear_Iter());

  unsigned long idx = 0;
  for( auto iNodes = 0ul; iNodes < ControlNodes.size(); iNodes++){
    /*--- Loop over the control nodes ---*/
    for (auto iNode = 0ul; iNode < ControlNodes[iNodes]->size(); iNode++) {
      
      // /*--- Setting nonzero displacement of the moving markers, else setting zero displacement for static markers---*/
      // if (config->GetMarker_All_Moving((*ControlNodes[iNodes])[iNode]->GetMarker())) {    
        

      //   for (auto iDim = 0u; iDim < nDim; iDim++) {
      //     CtrlNodeDeformation[idx++] = SU2_TYPE::GetValue(geometry->vertex[(*ControlNodes[iNodes])[iNode]->GetMarker()][(*ControlNodes[iNodes])[iNode]->GetVertex()]->GetVarCoord()[iDim] * VarIncrement);
      //   }
      // }
      
      // else{
      //   for (auto iDim = 0u; iDim < nDim; iDim++) {
      //     CtrlNodeDeformation[idx++] = 0.0;
      //     cout << config->GetMarker_All_TagBound((*ControlNodes[iNodes])[iNode]->GetMarker()) << endl;
      //     cout << SU2_TYPE::GetValue(geometry->vertex[(*ControlNodes[iNodes])[iNode]->GetMarker()][(*ControlNodes[iNodes])[iNode]->GetVertex()]->GetVarCoord()[iDim] * VarIncrement) << endl;
      //   }
      // }
      auto var_coord = geometry->vertex[(*ControlNodes[iNodes])[iNode]->GetMarker()][(*ControlNodes[iNodes])[iNode]->GetVertex()]->GetVarCoord();
      
      for ( auto iDim = 0u; iDim < nDim; iDim++ ){
        CtrlNodeDeformation[idx++] = var_coord[iDim] * VarIncrement;
      }      
    }
  }
  
  /*--- In case of a parallel computation, the deformation of all control nodes is send to the master process ---*/
  #ifdef HAVE_MPI
    
    /*--- Local number of control nodes ---*/
    // unsigned long Local_nControlNodes = ControlNodes->size(); //TODO can be replaced by nControlNodesLocal

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

  /*--- Looping over all nodes and check if part of domain and not on boundary ---*/
  for (auto iNode = 0ul; iNode < geometry->GetnPoint(); iNode++) {    
    if (!geometry->nodes->GetBoundary(iNode)) {
      internalNodes.push_back(iNode);
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
            if (find_if (SlideSurfNodes.begin(), SlideSurfNodes.end(), [&](CRadialBasisFunctionNode* i){return i->GetIndex() == iNode;}) == SlideSurfNodes.end()) {
              internalNodes.push_back(iNode);
            }
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
  if(config->GetRBF_DataReduction() && (BoundNodes.size() > 0 || SlideSurfNodes.size() > 0)){
    SetCorrection(geometry, config, type, internalNodes);  //TODO include the sliding nodes
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

    for(auto iNode = 0ul; iNode < SlideSurfNodes.size(); iNode++){
      
      /*--- Finding contribution of each control node ---*/
      for( auto jNode = 0ul; jNode <  nCtrlNodesGlobal; jNode++){
        
        /*--- Distance of non-selected boundary node to control node ---*/
        auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(SlideSurfNodes[iNode]->GetIndex()));
        
        /*--- Evaluation of the radial basis function based on the distance ---*/
        auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));

        /*--- Computing and add the resulting coordinate variation ---*/
        for(auto iDim = 0u; iDim < nDim; iDim++){
          var_coord[iDim] += rbf*InterpCoeff[jNode * nDim + iDim];
        }
      }

      /*--- Applying the coordinate variation and resetting the var_coord vector*/
      for(auto iDim = 0u; iDim < nDim; iDim++){
        geometry->nodes->AddCoord(SlideSurfNodes[iNode]->GetIndex(), iDim, var_coord[iDim]);
        var_coord[iDim] = 0;
      }
    }
  }

  // for(auto iNode = 0ul; iNode < SlideSurfNodes.size(); iNode++){//TODO check this out
      
  //   /*--- Finding contribution of each control node ---*/
  //   for( auto jNode = 0ul; jNode <  nCtrlNodesGlobal; jNode++){
      
  //     /*--- Distance of non-selected boundary node to control node ---*/
  //     auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(SlideSurfNodes[iNode]->GetIndex()));
      
  //     /*--- Evaluation of the radial basis function based on the distance ---*/
  //     auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));

  //     /*--- Computing and add the resulting coordinate variation ---*/
  //     for(auto iDim = 0u; iDim < nDim; iDim++){
  //       var_coord[iDim] += rbf*InterpCoeff[jNode * nDim + iDim];
  //     }
  //   }

  //   /*--- Applying the coordinate variation and resetting the var_coord vector*/
  //   for(auto iDim = 0u; iDim < nDim; iDim++){
  //     geometry->nodes->AddCoord(SlideSurfNodes[iNode]->GetIndex(), iDim, var_coord[iDim]);
  //     var_coord[iDim] = 0;
  //   }
  // }
  
  
  /*--- Applying the surface deformation, which are stored in the deformation vector ---*/
  // for(auto jNode = 0ul; jNode < ControlNodes[0]->size(); jNode++){ 
  //   cout << (*ControlNodes[0])[jNode]->GetIndex() << endl;
  //   if(config->GetMarker_All_Moving((*ControlNodes[0])[jNode]->GetMarker())){
  //     for(auto iDim = 0u; iDim < nDim; iDim++){
  //         geometry->nodes->AddCoord((*ControlNodes[0])[jNode]->GetIndex(), iDim, CtrlNodeDeformation[jNode*nDim + iDim]); 
  //     }
  //   }
  // } 
  ofstream myfile;
  myfile.open ("appliedDeformation.txt");

  unsigned long idx = 0;
  for( auto iNodes = 0ul; iNodes < ControlNodes.size(); iNodes++){
    for(auto jNode = 0ul; jNode < ControlNodes[iNodes]->size(); jNode++){ 
      myfile << jNode << "\t" << CtrlNodeDeformation[idx] << "\t" << CtrlNodeDeformation[idx+1] << "\t" << CtrlNodeDeformation[idx+2] << endl;
      for(auto iDim = 0u; iDim < nDim; iDim++){
          geometry->nodes->AddCoord((*ControlNodes[iNodes])[jNode]->GetIndex(), iDim, CtrlNodeDeformation[idx++]); 
      }
    }
  }
  
  
  myfile.close();
}


void CRadialBasisFunctionInterpolation::GetInitMaxErrorNode(CGeometry* geometry, CConfig* config, vector<CRadialBasisFunctionNode*>& maxErrorNodes, su2double& maxErrorLocal){

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
    maxErrorNodes.push_back(BoundNodes[maxErrorNodeLocal]);

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
  su2double localCoords[nDim*nCtrlNodesLocal];
  
  /*--- Storing local control node coordinates ---*/
  unsigned long idx = 0;
  for(auto iNodes = 0u; iNodes < ControlNodes.size(); iNodes++){
    for(auto iNode = 0ul; iNode < ControlNodes[iNodes]->size(); iNode++){
      auto coord = geometry->nodes->GetCoord((*ControlNodes[iNodes])[iNode]->GetIndex());  
      for ( auto iDim = 0u ; iDim < nDim; iDim++ ){
        localCoords[ idx++ ] = coord[iDim];
      }
    }
  }

  /*--- Gathering local control node coordinate sizes on all processes. ---*/
  int LocalCoordsSizes[size];
  int localCoordsSize = nDim*nCtrlNodesLocal;
  SU2_MPI::Allgather(&localCoordsSize, 1, MPI_INT, LocalCoordsSizes, 1, MPI_INT, SU2_MPI::GetComm()); 

  /*--- Array containing the starting indices for the allgatherv operation */
  int disps[SU2_MPI::GetSize()] = {0};    

  for(auto iProc = 1; iProc < SU2_MPI::GetSize(); iProc++){
    disps[iProc] = disps[iProc-1]+LocalCoordsSizes[iProc-1];
  }
  
  /*--- Distributing global control node coordinates among all processes ---*/
  SU2_MPI::Allgatherv(&localCoords, localCoordsSize, MPI_DOUBLE, CtrlCoords.data(), LocalCoordsSizes, disps, MPI_DOUBLE, SU2_MPI::GetComm()); 

};


void CRadialBasisFunctionInterpolation::GetInterpError(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, su2double& maxErrorLocal, vector<CRadialBasisFunctionNode*>& maxErrorNodes){
  
  /*--- Array containing the local error ---*/
  su2double localError[nDim];

  /*--- Magnitude of the local maximum error ---*/
  maxErrorLocal = 0.0;
  CRadialBasisFunctionNode* maxErrorNodeLocal;
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
      maxErrorNodeLocal = BoundNodes[iNode];
    }
  }

  if(SlideSurfNodes.size() > 0){
    /*--- Vector storing the coordinates of the boundary nodes ---*/
    vector<su2double> Coord_bound;

    /*--- Vector storing the IDs of the boundary nodes ---*/
    vector<unsigned long> PointIDs;

    unsigned long idx = 0;
    // for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++) {
    //   if(config->GetMarker_All_Deform_Mesh_Slide(iMarker)){
    for ( auto iVertex = 0ul; iVertex < SlideSurfNodes.size(); iVertex++){

      // auto iNode = geometry->vertex[iMarker][iVertex]->GetNode();
      auto iNode = SlideSurfNodes[iVertex]->GetIndex();
      PointIDs.push_back(idx++); //TODO
      for(auto iDim = 0u; iDim < nDim; iDim++){
        Coord_bound.push_back(geometry->nodes->GetCoord(iNode, iDim));
      }
      // cout << iNode << " " << Coord_bound[Coord_bound.size()-3] << " " << Coord_bound[Coord_bound.size()-2] << " " << Coord_bound[Coord_bound.size()-1] << endl;
    }

    if(ControlNodes.size() > 1){
      for(auto iVertex = 0ul; iVertex < ControlNodes[1]->size(); iVertex++){
        auto iNode = (*ControlNodes[1])[iVertex]->GetIndex();
        PointIDs.push_back(idx++); //TODO
        for(auto iDim = 0u; iDim < nDim; iDim++){
          Coord_bound.push_back(geometry->nodes->GetCoord(iNode, iDim));
        }
      }
    }
    
    /*--- Number of non-selected boundary nodes ---*/
    const unsigned long nVertexBound = PointIDs.size();

    /*--- Construction of AD tree ---*/
    CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);
    
    for (auto iNode = 0ul; iNode < SlideSurfNodes.size(); iNode++){
      GetSlidingNodalError(geometry, config, type, radius, iNode, localError, BoundADT);

      /*--- Setting error ---*/
      SlideSurfNodes[iNode]->SetError(localError, nDim);

      /*--- Compute error magnitude and update local maximum error if necessary ---*/
      su2double errorMagnitude = GeometryToolbox::Norm(nDim, localError);
      if(errorMagnitude > maxErrorLocal){
        maxErrorLocal = errorMagnitude;
        maxErrorNodeLocal = SlideSurfNodes[iNode];
      }
    }  
  }

  if(maxErrorLocal > 0){
    /*--- Including the maximum error nodes in the max error nodes vector ---*/
    maxErrorNodes.push_back(maxErrorNodeLocal);
    
    /*--- Finding a double edged error node if possible ---*/
    GetDoubleEdgeNode(maxErrorNodes[0]->GetError(), maxErrorNodes);
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

void CRadialBasisFunctionInterpolation::GetDoubleEdgeNode(const su2double* maxError, vector<CRadialBasisFunctionNode*>& maxErrorNodes){
  
  /*--- Obtaining maximum error vector and its corresponding angle ---*/
  const auto polarAngleMaxError = atan2(maxError[1],maxError[0]);
  const auto azimuthAngleMaxError = atan2( sqrt( pow(maxError[0],2) + pow(maxError[1],2)), maxError[2]);

  su2double max = 0;
  CRadialBasisFunctionNode* idx = nullptr;
  bool found = false;

  for(auto iNode = 0ul; iNode < BoundNodes.size(); iNode++){
    
    auto error = BoundNodes[iNode]->GetError();
    su2double polarAngle = atan2(error[1],error[0]);
    su2double relativePolarAngle = abs(polarAngle - polarAngleMaxError);

    switch(nDim){
      case 2:
        if ( abs(relativePolarAngle - M_PI) < M_PI/2 ){
          CompareError(error, BoundNodes[iNode], max, &idx);
        }
        break;
      case 3:

        su2double azimuthAngle = atan2( sqrt( pow(error[0],2) + pow(error[1],2)), error[2]);
        if ( abs(relativePolarAngle - M_PI) < M_PI/2 ){
          if( azimuthAngleMaxError <= M_PI/2 ) {
            if ( azimuthAngle > M_PI/2 - azimuthAngleMaxError) {
              CompareError(error, BoundNodes[iNode], max, &idx);
            }
          }
          else{
            if ( azimuthAngle < 1.5 * M_PI - azimuthAngleMaxError ){
              CompareError(error, BoundNodes[iNode], max, &idx);
            }
          }          
        }
        else{
          if(azimuthAngleMaxError <= M_PI/2){
            if(azimuthAngle > M_PI/2 + azimuthAngleMaxError){
              CompareError(error, BoundNodes[iNode], max, &idx);
            }
          }else{
            if(azimuthAngle < azimuthAngleMaxError -  M_PI/2){
              CompareError(error, BoundNodes[iNode], max, &idx);
            }
          }
        }
        break;
    }
  }

  // for(auto iNode = 0ul; iNode < SlideSurfNodes.size(); iNode++){
    
  //   auto error = SlideSurfNodes[iNode]->GetError();
  //   su2double polarAngle = atan2(error[1],error[0]);
  //   su2double relativePolarAngle = abs(polarAngle - polarAngleMaxError);

  //   switch(nDim){
  //     case 2:
  //       if ( abs(relativePolarAngle - M_PI) < M_PI/2 ){
  //         CompareError(error, iNode, max, idx);
  //       }
  //       break;
  //     case 3:

  //       su2double azimuthAngle = atan2( sqrt( pow(error[0],2) + pow(error[1],2)), error[2]);
  //       if ( abs(relativePolarAngle - M_PI) < M_PI/2 ){
  //         if( azimuthAngleMaxError <= M_PI/2 ) {
  //           if ( azimuthAngle > M_PI/2 - azimuthAngleMaxError) {
  //             CompareError(error, iNode, max, idx);
  //           }
  //         }
  //         else{
  //           if ( azimuthAngle < 1.5 * M_PI - azimuthAngleMaxError ){
  //             CompareError(error, iNode, max, idx);
  //           }
  //         }          
  //       }
  //       else{
  //         if(azimuthAngleMaxError <= M_PI/2){
  //           if(azimuthAngle > M_PI/2 + azimuthAngleMaxError){
  //             CompareError(error, iNode, max, idx);
  //           }
  //         }else{
  //           if(azimuthAngle < azimuthAngleMaxError -  M_PI/2){
  //             CompareError(error, iNode, max, idx);
  //           }
  //         }
  //       }
  //       break;
  //   }
  // }
  

  /*--- Include the found double edge node in the maximum error nodes vector ---*/
  if( idx != nullptr){
    maxErrorNodes.push_back(idx);
  }
}

void CRadialBasisFunctionInterpolation::CompareError(su2double* error, CRadialBasisFunctionNode* iNode, su2double& maxError, CRadialBasisFunctionNode** idx){
  auto errMag = GeometryToolbox::SquaredNorm(nDim, error);

  if(errMag > maxError){
    maxError = errMag;
    *idx = iNode;
  }
}

void CRadialBasisFunctionInterpolation::SetCorrection(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const vector<unsigned long>& internalNodes){

  /*--- The non-selected control nodes still have a nonzero error once the maximum error falls below the data reduction tolerance. 
          This error is applied as correction and interpolated into the volumetric mesh for internal nodes that fall within the correction radius.
          To evaluate whether an internal node falls within the correction radius an AD tree is constructed of the boundary nodes,
          making it possible to determine the distance to the nearest boundary node. ---*/

  /*--- Construction of the AD tree consisting of the non-selected boundary nodes ---*/

  /*--- Number of non-selected boundary nodes ---*/
  const unsigned long nVertexBound = BoundNodes.size() + SlideSurfNodes.size();
  
  /*--- Vector storing the coordinates of the boundary nodes ---*/
  vector<su2double> Coord_bound(nDim*nVertexBound);

  /*--- Vector storing the IDs of the boundary nodes ---*/
  vector<unsigned long> PointIDs(nVertexBound);

  /*--- Correction Radius, equal to maximum error times a prescribed constant ---*/
  const su2double CorrectionRadius = config->GetRBF_DataRedCorrectionFactor()*MaxErrorGlobal; 

  /*--- Storing boundary node information ---*/
  unsigned long i = 0;
  unsigned long j = 0;
  for(auto iVertex = 0ul; iVertex < BoundNodes.size(); iVertex++){
    auto iNode = BoundNodes[iVertex]->GetIndex();
    PointIDs[i++] = iVertex;
    for(auto iDim = 0u; iDim < nDim; iDim++){
      Coord_bound[j++] = geometry->nodes->GetCoord(iNode, iDim);
    }
  }

  //sliding surf nodes
  for(auto iVertex = 0ul; iVertex < SlideSurfNodes.size(); iVertex++){
    auto iNode = SlideSurfNodes[iVertex]->GetIndex();
    PointIDs[i++] = iVertex+BoundNodes.size();
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
    CRadialBasisFunctionNode* nearestNode;
    if( pointID < BoundNodes.size()){
      nearestNode = BoundNodes[pointID];
    }else{
      nearestNode = SlideSurfNodes[pointID - BoundNodes.size()];
    }
    auto err = nearestNode->GetError();

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

  for(auto iNode = 0ul; iNode < SlideSurfNodes.size(); iNode++){
    auto err =  SlideSurfNodes[iNode]->GetError();
    for(auto iDim = 0u; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(SlideSurfNodes[iNode]->GetIndex(), iDim, -err[iDim]);
    }
  }
}


void CRadialBasisFunctionInterpolation::AddControlNode(vector<CRadialBasisFunctionNode*>& maxErrorNodes){
  /*--- Sort indices in descending order, to prevent shift in index as they are erased ---*/
  sort(maxErrorNodes.rbegin(),maxErrorNodes.rend(), HasSmallerIndex); 
  // check which type of node
  for(auto iNode : maxErrorNodes){
    if(find(BoundNodes.begin(), BoundNodes.end(), iNode) != BoundNodes.end()){
      
      /*--- Addition of node to the reduced set of control nodes ---*/
      ReducedBoundNodes.push_back(move(iNode));

      /*--- Removal of node among the non-selected boundary nodes ---*/
      BoundNodes.erase( remove(BoundNodes.begin(), BoundNodes.end(), iNode), BoundNodes.end());
      // BoundNodes.erase(BoundNodes.begin()+iNode);
    
    }else{
      /*--- Addition of node to the reduced set of control nodes ---*/
      ReducedSlideNodes.push_back(move(iNode));

      /*--- Removal of node among the non-selected boundary nodes ---*/
      SlideSurfNodes.erase( remove(SlideSurfNodes.begin(), SlideSurfNodes.end(), iNode), SlideSurfNodes.end());
    }
  }
    

  /*--- Clearing maxErrorNodes vector ---*/
  maxErrorNodes.clear();
}


void CRadialBasisFunctionInterpolation::Get_nCtrlNodesGlobal(){
  /*--- Determining the global number of control nodes ---*/
  
  /*--- Local number of control nodes ---*/
  nCtrlNodesLocal = 0;

  for(auto iNodes = 0u; iNodes < ControlNodes.size(); iNodes++){
    nCtrlNodesLocal += ControlNodes[iNodes]->size();
  }

  /*--- Summation of local number of control nodes ---*/
  SU2_MPI::Allreduce(&nCtrlNodesLocal, &nCtrlNodesGlobal, 1, MPI_UNSIGNED_LONG, MPI_SUM, SU2_MPI::GetComm());

}

void CRadialBasisFunctionInterpolation::GetSlidingDeformation(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, CADTPointsOnlyClass& BoundADT){
  cout << "sliding surf calc" << endl;
  GetInterpCoeffs(geometry, config, type, radius);
  
  su2double new_coord[nDim]{0.0};

  /*--- ID of nearest boundary node ---*/
  unsigned long pointID;
  /*--- Distance to nearest boundary node ---*/
  su2double dist;
  /*--- rank of nearest boundary node ---*/
  int rankID;



  for(auto iNode = 0ul; iNode < SlideSurfNodes.size(); iNode++){ //TODO nodes here should be SlideSurfNodes/reducedSlideNodes

    auto coord = geometry->nodes->GetCoord(SlideSurfNodes[iNode]->GetIndex());

    for(auto iDim = 0u; iDim < nDim; iDim++){
      new_coord[iDim] = coord[iDim];
    }


    /*--- Finding contribution of each control node ---*/
    for( auto jNode = 0ul; jNode <  nCtrlNodesGlobal; jNode++){
      
      /*--- Distance of non-selected boundary node to control node ---*/
      auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(SlideSurfNodes[iNode]->GetIndex()));
      
      /*--- Evaluation of the radial basis function based on the distance ---*/
      auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));

      /*--- Computing and add the resulting coordinate variation ---*/
      for(auto iDim = 0u; iDim < nDim; iDim++){
        new_coord[iDim] += rbf*InterpCoeff[jNode * nDim + iDim];
      }
    }

    
    

    BoundADT.DetermineNearestNode(new_coord, dist, pointID, rankID);

    unsigned short marker;
    unsigned long vertex;
    if(pointID < SlideSurfNodes.size()){
      marker = SlideSurfNodes[pointID]->GetMarker();
      vertex = SlideSurfNodes[pointID]->GetVertex();
    }else{
      marker = ReducedSlideNodes[pointID-SlideSurfNodes.size()]->GetMarker();
      vertex = ReducedSlideNodes[pointID-SlideSurfNodes.size()]->GetVertex();
    }
    // geometry->vertex[SlideSurfNodes[]]
    // cout << iNode << " " << SlideSurfNodes[iNode]->GetIndex() << " Nearest node: " << SlideSurfNodes[pointID]->GetIndex() << endl;
    auto normal =  geometry->vertex[marker][vertex]->GetNormal();
    // cout << normal[0] << " " << normal[1] <<" " << normal[2] << endl;
    
    su2double dist_vec[nDim];
    GeometryToolbox::Distance(nDim, new_coord, geometry->nodes->GetCoord(geometry->vertex[marker][vertex]->GetNode()), dist_vec);

    auto dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

    auto norm_magnitude =  GeometryToolbox::Norm(nDim, normal);
    su2double var_coord[nDim];

    for(auto iDim = 0u; iDim < nDim; iDim++){
      new_coord[iDim] += -dot_product*normal[iDim]/pow(norm_magnitude,2);
      var_coord[iDim] = (new_coord[iDim] - coord[iDim])*((su2double)config->GetGridDef_Nonlinear_Iter());
    }


    geometry->vertex[SlideSurfNodes[iNode]->GetMarker()][SlideSurfNodes[iNode]->GetVertex()]->SetVarCoord(var_coord); 
    
    /*--- Applying the coordinate variation and resetting the var_coord vector*/
    // for(auto iDim = 0u; iDim < nDim; iDim++){
    //   geometry->nodes->AddCoord(SlideSurfNodes[iNode]->GetIndex(), iDim, var_coord[iDim]);
    //   var_coord[iDim] = 0;
    // }
    
  }
  // exit(0);
  // BoundNodes.insert(BoundNodes.end(), SlideSurfNodes.begin(), SlideSurfNodes.end());
}


// CADTPointsOnlyClass CRadialBasisFunctionInterpolation::GetADT(CGeometry* geometry){
  
//   // return BoundADT;
// }


void CRadialBasisFunctionInterpolation::GetSlidingNodalError(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, unsigned long iNode, su2double* localError, CADTPointsOnlyClass& BoundADT){
  auto coord = geometry->nodes->GetCoord(SlideSurfNodes[iNode]->GetIndex());
  
  for(auto iDim = 0u; iDim < nDim; iDim++){
    localError[iDim] = coord[iDim]; 
  }

  for(auto jNode = 0ul; jNode < nCtrlNodesGlobal; jNode++){

    /*--- Distance between non-selected boundary node and control node ---*/
    auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode *nDim], geometry->nodes->GetCoord(SlideSurfNodes[iNode]->GetIndex()));

    /*--- Evaluation of Radial Basis Function ---*/
    auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));

    /*--- Add contribution to error ---*/
    for(auto iDim = 0u; iDim < nDim; iDim++){
      localError[iDim] += rbf*InterpCoeff[jNode*nDim + iDim];
    }
  }


  /*--- ID of nearest boundary node ---*/
  unsigned long pointID;
  /*--- Distance to nearest boundary node ---*/
  su2double dist;
  /*--- rank of nearest boundary node ---*/
  int rankID;

  BoundADT.DetermineNearestNode(localError, dist, pointID, rankID);  
  

  // su2double* normal;
  unsigned short marker;
  unsigned long vertex;
  if(pointID < SlideSurfNodes.size()){
    marker = SlideSurfNodes[pointID]->GetMarker();
    vertex = SlideSurfNodes[pointID]->GetVertex();
    
  }else{
    marker = (*ControlNodes[1])[pointID-SlideSurfNodes.size()]->GetMarker();
    vertex = (*ControlNodes[1])[pointID-SlideSurfNodes.size()]->GetVertex();
    
  }

  auto normal = geometry->vertex[marker][vertex]->GetNormal();
  // geometry->nodes->Get
  // auto normal =  geometry->vertex[SlideSurfNodes[pointID]->GetMarker()][SlideSurfNodes[pointID]->GetVertex()]->GetNormal();
  // // cout << normal[0] << " " << normal[1] <<" " << normal[2] << endl;
  
  su2double dist_vec[nDim];
  GeometryToolbox::Distance(nDim, localError, geometry->nodes->GetCoord(geometry->vertex[marker][vertex]->GetNode()), dist_vec);
  auto norm_magnitude =  GeometryToolbox::Norm(nDim, normal);
  
  auto dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec)/GeometryToolbox::Norm(nDim, normal);

  for( auto iDim = 0u; iDim < nDim; iDim++){
    // localError[iDim] = normal[iDim]*dist_vec[iDim]/norm_magnitude;
    localError[iDim] = dot_product*normal[iDim]/GeometryToolbox::Norm(nDim, normal);
  }
  // 
  
  // auto norm_magnitude =  GeometryToolbox::Norm(nDim, normal);
  
  
  // for(auto iDim = 0u; iDim < nDim; iDim++){
  //   new_coord[iDim] += -dot_product*normal[iDim]/pow(norm_magnitude,2);
    
  // }
}