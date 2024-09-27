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

  //TODO do elsewhere
  red_wall.resize(config->GetnMarker_Deform_Mesh_IL_Wall());
  for (auto iMarker = 0u; iMarker < config->GetnMarker_Deform_Mesh_IL_Wall(); iMarker++){
    red_wall[iMarker] = new vector<CRadialBasisFunctionNode*>;
  }
  red_edge.resize(config->GetnMarker_Deform_Mesh_IL_Wall());
  for (auto iMarker = 0u; iMarker < config->GetnMarker_Deform_Mesh_IL_Wall(); iMarker++){
    red_edge[iMarker] = new vector<CRadialBasisFunctionNode*>;
  }

  /*--- Looping over the number of deformation iterations ---*/
  for (auto iNonlinear_Iter = 0ul; iNonlinear_Iter < Nonlinear_Iter; iNonlinear_Iter++) {
    
    /*--- Compute min volume in the entire mesh. ---*/

    ComputeDeforming_Element_Volume(geometry, MinVolume, MaxVolume, Screen_Output);

    if (rank == MASTER_NODE && Screen_Output)
      cout << "Min. volume: " << MinVolume << ", max. volume: " << MaxVolume << "." << endl;
    

    /*--- Solving the RBF system, resulting in the interpolation coefficients ---*/
    SolveRBF_System(geometry, config, kindRBF, radius);

    /*--- Updating the coordinates of the grid ---*/
    UpdateGridCoord(geometry, config, kindRBF, radius, internalNodes, false);
    
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
    vector<unsigned long> maxErrorNodes;
    vector<unsigned short> maxErrorVector;
    
    su2double maxErrorLocal{0.};
    // unsigned short maxErrorVector{0};

    if(config->GetRBF_IL_Preservation()){

      for ( auto iLayer = 0u; iLayer < config->GetnMarker_Deform_Mesh_IL_Wall(); iLayer++){
        // setting control and boundary nodes, put input vectors at first index
        SetNodes(red_wall[iLayer], test[iLayer], 0);

        // Resetting error data
        ResetError(maxErrorNodes, maxErrorVector, maxErrorLocal);
        
        // Get initial node based on max deformation
        if(nCtrlNodesGlobal == 0){
          GetInitMaxErrorNode(geometry, config, *test[iLayer], maxErrorNodes, maxErrorVector, maxErrorLocal);
          SU2_MPI::Allreduce(&maxErrorLocal, &MaxErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());

          /*--- Error tolerance for the data reduction tolerance ---*/
        DataReductionTolerance = MaxErrorGlobal*config->GetRBF_DataRedTolerance() * ((su2double)config->GetGridDef_Nonlinear_Iter());
        DataRedTol_IL = MaxErrorGlobal*config->GetRBF_IL_DataRedTolerance() * ((su2double)config->GetGridDef_Nonlinear_Iter());
        }

        // nr of iters
        auto iter_wall = 0u;

        // do iterations 
        while(MaxErrorGlobal > DataRedTol_IL || iter_wall == 0){
          
          if(maxErrorLocal > 0){
            AddControlNode(maxErrorVector, maxErrorNodes, maxErrorLocal);
          }

          GetInterpCoeffs(geometry, config, type, radius);
          
          GetInterpError(geometry, config, type, radius, maxErrorLocal, maxErrorNodes, maxErrorVector);
          SU2_MPI::Allreduce(&maxErrorLocal, &MaxErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());

          cout.precision(5);
          iter_wall++;

          if (rank == MASTER_NODE && iter_wall % 10 == 0){
              cout << "Wall iter nr: " << iter_wall << ",\t Nr. of ctrl nodes: " << nCtrlNodesGlobal <<  ",\t error: " << MaxErrorGlobal << ",\t tol: " << DataRedTol_IL << endl;
          }
        }

        // TODO should be together. 
        GetIL_EdgeDeformation(geometry, config, type, radius, iLayer);

        SetNodes(red_edge[iLayer], test_edge[iLayer], 1);

        ResetError(maxErrorNodes, maxErrorVector, maxErrorLocal);

        auto iter = 0u;

        while(MaxErrorGlobal > DataRedTol_IL || iter == 0){

          if(maxErrorLocal > 0){
            AddControlNode(maxErrorVector, maxErrorNodes, maxErrorLocal);
          }

          GetInterpCoeffs(geometry, config, type, radius);
          GetInterpError(geometry, config, type, radius, maxErrorLocal, maxErrorNodes, maxErrorVector);
          SU2_MPI::Allreduce(&maxErrorLocal, &MaxErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());

          iter++;

          if (rank == MASTER_NODE && iter % 10 == 0){
            cout << "Edge iter nr: " << iter << ",\t Nr. of ctrl nodes: " << nCtrlNodesGlobal <<  ",\t error: " << MaxErrorGlobal << ",\t tol: " << DataRedTol_IL << endl;
          }
        }
        
        /*--- Updating the inflation layer coordinates ---*/ 
        UpdateGridCoord(geometry, config, type, radius, *IL_internalNodes[iLayer], true);
      }


      //TODO remove these check files
      ofstream of;
      of.open("reducedwall.txt");
      // for(auto x : Reduced_IL_WallNodes){ of << x->GetIndex() << endl;}
      for (auto x : red_wall){
        for (auto y : *x){
          of << y->GetIndex() << endl;
        }
      }
      of.close();

      of.open("reducededge.txt");
      // for(auto x : Reduced_IL_EdgeNodes){ of << x->GetIndex() << endl;}
      for (auto x : red_edge){
        for (auto y : *x){
          of << y->GetIndex() << endl;
        }
      }
      of.close();
      }

      // some setControlNode function
      if(config->GetRBF_IL_Preservation()){ 
        /*--- Set control nodes for the domain outside inflation layer ---*/
        ControlNodes.resize(config->GetnMarker_Deform_Mesh_IL_Wall() + 1); // reduced other boundary nodes and the edges of the inflation layers
        BdryNodes.resize(config->GetnMarker_Deform_Mesh_IL_Wall() + 1);

        ControlNodes[0] = &ReducedControlNodes;
        BdryNodes[0] = &BoundNodes;

        for (auto iLayer = 0u; iLayer < config->GetnMarker_Deform_Mesh_IL_Wall(); iLayer++){
          ControlNodes[iLayer + 1] = red_edge[iLayer];
          BdryNodes[iLayer + 1] = test_edge[iLayer];
        }
        Get_nCtrlNodes();

        
      }else{
        BdryNodes.resize(1);
        BdryNodes[0]= &BoundNodes;
        types.resize(1);
        types[0] = "MOVING";
        
      }

      ResetError(maxErrorNodes, maxErrorVector, maxErrorLocal);
      // maxErrorLocal = 0;
      // maxErrorNodes.clear();

      auto iter_domain = 0u;    

      // in case no inflation layer iterations were done a boundNode has to be selected
      if(nCtrlNodesGlobal == 0){
        GetInitMaxErrorNode(geometry, config, BoundNodes, maxErrorNodes, maxErrorVector, maxErrorLocal);
        SU2_MPI::Allreduce(&maxErrorLocal, &MaxErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());

        /*--- Error tolerance for the data reduction tolerance ---*/
        if(MaxErrorGlobal != 0){
          DataReductionTolerance = MaxErrorGlobal*config->GetRBF_DataRedTolerance() * ((su2double)config->GetGridDef_Nonlinear_Iter());
        }
      }

      while(MaxErrorGlobal > DataReductionTolerance || iter_domain == 0){

        if(maxErrorLocal > 0){
          AddControlNode(maxErrorVector, maxErrorNodes, maxErrorLocal);
        }
        ControlNodes.resize(1);
        BdryNodes.resize(1);
        Get_nCtrlNodes();
        types.resize(1);
        types[0] = "MOVING";

        if(red_SlideEdgeNodes.size() > 0){
          
          GetInterpCoeffs(geometry, config, type, radius);

          // construction ad tree
          
          /*--- Vector storing the coordinates of the boundary nodes ---*/
          vector<su2double> Coord_bound((red_SlideEdgeNodes.size()+SlideEdgeNodes.size()) * nDim);

          /*--- Vector storing the IDs of the boundary nodes ---*/
          vector<unsigned long> PointIDs((red_SlideEdgeNodes.size()+SlideEdgeNodes.size()));

          for ( auto iVertex = 0ul; iVertex < SlideEdgeNodes.size(); iVertex++){

            // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
            PointIDs[iVertex] = SlideEdgeNodes[iVertex]->GetVertex(); 
            for(auto iDim = 0u; iDim < nDim; iDim++){
              Coord_bound[ iVertex*nDim + iDim]  = geometry->nodes->GetCoord(SlideEdgeNodes[iVertex]->GetIndex(), iDim);
            }
          }

          for ( auto iVertex = 0ul; iVertex < red_SlideEdgeNodes.size(); iVertex++){

            // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
            PointIDs[iVertex + SlideEdgeNodes.size()] = red_SlideEdgeNodes[iVertex]->GetVertex();  
            for(auto iDim = 0u; iDim < nDim; iDim++){
              Coord_bound[ (SlideEdgeNodes.size() + iVertex) * nDim + iDim]  = geometry->nodes->GetCoord(red_SlideEdgeNodes[iVertex]->GetIndex(), iDim);
            }
          }
          
          /*--- Number of non-selected boundary nodes ---*/
          const unsigned long nVertexBound = PointIDs.size();

          /*--- Construction of AD tree ---*/
          CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);
          
          
          // free deformation
          
          GetFreeDeformation(geometry, config, type, radius, &red_SlideEdgeNodes);
          
          /*--- ID of nearest boundary node ---*/
          unsigned long pointID;
          /*--- Distance to nearest boundary node ---*/
          su2double dist;
          /*--- rank of nearest boundary node ---*/
          int rankID;

          // loop through the sliding nodes 
          for(auto iNode : red_SlideEdgeNodes){
            BoundADT.DetermineNearestNode(iNode->GetNewCoord(), dist, pointID, rankID);

            // geometry->vertex[SlideSurfNodes[]]
            // cout << iNode << " " << SlideSurfNodes[iNode]->GetIndex() << " Nearest node: " << SlideSurfNodes[pointID]->GetIndex() << endl;
            auto normal =  geometry->vertex[iNode->GetMarker()][pointID]->GetNormal();
            // cout << normal[0] << " " << normal[1] <<" " << normal[2] << endl;
            
            su2double dist_vec[nDim];
            GeometryToolbox::Distance(nDim, iNode->GetNewCoord(), geometry->nodes->GetCoord(geometry->vertex[iNode->GetMarker()][pointID]->GetNode()), dist_vec);

            auto dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

            auto norm_magnitude =  GeometryToolbox::Norm(nDim, normal);


            for(auto iDim = 0u; iDim < nDim; iDim++){
              iNode->AddNewCoord(-dot_product*normal[iDim]/pow(norm_magnitude,2), iDim);
            }

            if(nDim == 3 && !geometry->nodes->GetPeriodicBoundary(iNode->GetIndex())){
              // there is a second surface for this edge
              unsigned short mark; 
              for( auto i = 0u; i < config->GetnMarker_All(); i++){
                if(geometry->nodes->GetVertex(geometry->vertex[iNode->GetMarker()][pointID]->GetNode(), i) != -1 && i != iNode->GetMarker()){
                  mark = i;
                  break;
                }
              }

              auto jNode = geometry->nodes->GetVertex(geometry->vertex[iNode->GetMarker()][pointID]->GetNode(), mark);
              if(jNode != -1){
                normal = geometry->vertex[mark][jNode]->GetNormal();
              
                dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

                norm_magnitude =  GeometryToolbox::Norm(nDim, normal);
                
                for(auto iDim = 0u; iDim < nDim; iDim++){
                  iNode->AddNewCoord(-dot_product*normal[iDim]/pow(norm_magnitude,2), iDim);
                }
              }
            }

            su2double var_coord[nDim];

            for(auto iDim = 0u; iDim < nDim; iDim++){
              var_coord[iDim] = (iNode->GetNewCoord()[iDim]  - geometry->nodes->GetCoord(iNode->GetIndex())[iDim])*((su2double)config->GetGridDef_Nonlinear_Iter()); //TODO  var_coord can likely be obtained better
            }

            geometry->vertex[iNode->GetMarker()][iNode->GetVertex()]->SetVarCoord(var_coord); 

          }

          ControlNodes.push_back(&red_SlideEdgeNodes);
          BdryNodes.push_back(&SlideEdgeNodes);
          Get_nCtrlNodes();
          types.push_back("EDGE");

        }
        
        
        GetInterpCoeffs(geometry, config, type, radius);
        GetInterpError(geometry, config, type, radius, maxErrorLocal, maxErrorNodes, maxErrorVector);
        SU2_MPI::Allreduce(&maxErrorLocal, &MaxErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());

        iter_domain++;
        
        if (rank == MASTER_NODE && iter_domain % 1 == 0){
          cout << "domain iter nr: " << iter_domain << ",\t Nr. of ctrl nodes: " << nCtrlNodesGlobal <<  ",\t error: " << MaxErrorGlobal << ",\t tol: " << DataReductionTolerance << endl;
        }        
      }   

      if(SlideSurfNodes.size() > 0){
        ResetError(maxErrorNodes, maxErrorVector, maxErrorLocal);

        auto iter_surf = 0u;    

        while(MaxErrorGlobal > DataReductionTolerance || iter_surf == 0){

          if(maxErrorLocal > 0){
            AddControlNode(maxErrorVector, maxErrorNodes, maxErrorLocal);
          }

          // start with bound nodes as control nodes
          ControlNodes.resize(1);
          BdryNodes.resize(1);
          Get_nCtrlNodes();
          types.resize(1);
          types[0] = "MOVING";

          //finding coord var of the sliding edge nodes
          if(red_SlideEdgeNodes.size() > 0){
            
            GetInterpCoeffs(geometry, config, type, radius);

            // construction ad tree
            
            /*--- Vector storing the coordinates of the boundary nodes ---*/
            vector<su2double> Coord_bound((red_SlideEdgeNodes.size()+SlideEdgeNodes.size()) * nDim);

            /*--- Vector storing the IDs of the boundary nodes ---*/
            vector<unsigned long> PointIDs((red_SlideEdgeNodes.size()+SlideEdgeNodes.size()));

            for ( auto iVertex = 0ul; iVertex < SlideEdgeNodes.size(); iVertex++){

              // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
              PointIDs[iVertex] = SlideEdgeNodes[iVertex]->GetVertex(); 
              for(auto iDim = 0u; iDim < nDim; iDim++){
                Coord_bound[ iVertex*nDim + iDim]  = geometry->nodes->GetCoord(SlideEdgeNodes[iVertex]->GetIndex(), iDim);
              }
            }

            for ( auto iVertex = 0ul; iVertex < red_SlideEdgeNodes.size(); iVertex++){

              // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
              PointIDs[iVertex + SlideEdgeNodes.size()] = red_SlideEdgeNodes[iVertex]->GetVertex();  
              for(auto iDim = 0u; iDim < nDim; iDim++){
                Coord_bound[ (SlideEdgeNodes.size() + iVertex) * nDim + iDim]  = geometry->nodes->GetCoord(red_SlideEdgeNodes[iVertex]->GetIndex(), iDim);
              }
            }
            
            /*--- Number of non-selected boundary nodes ---*/
            const unsigned long nVertexBound = PointIDs.size();

            /*--- Construction of AD tree ---*/
            CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);
            
            
            // free deformation
            
            GetFreeDeformation(geometry, config, type, radius, &red_SlideEdgeNodes);
            
            /*--- ID of nearest boundary node ---*/
            unsigned long pointID;
            /*--- Distance to nearest boundary node ---*/
            su2double dist;
            /*--- rank of nearest boundary node ---*/
            int rankID;

            // loop through the sliding nodes 
            for(auto iNode : red_SlideEdgeNodes){
              BoundADT.DetermineNearestNode(iNode->GetNewCoord(), dist, pointID, rankID);

              // geometry->vertex[SlideSurfNodes[]]
              // cout << iNode << " " << SlideSurfNodes[iNode]->GetIndex() << " Nearest node: " << SlideSurfNodes[pointID]->GetIndex() << endl;
              auto normal =  geometry->vertex[iNode->GetMarker()][pointID]->GetNormal();
              // cout << normal[0] << " " << normal[1] <<" " << normal[2] << endl;
              
              su2double dist_vec[nDim];
              GeometryToolbox::Distance(nDim, iNode->GetNewCoord(), geometry->nodes->GetCoord(geometry->vertex[iNode->GetMarker()][pointID]->GetNode()), dist_vec);

              auto dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

              auto norm_magnitude =  GeometryToolbox::Norm(nDim, normal);


              for(auto iDim = 0u; iDim < nDim; iDim++){
                iNode->AddNewCoord(-dot_product*normal[iDim]/pow(norm_magnitude,2), iDim);
              }

              if(nDim == 3 &&  !geometry->nodes->GetPeriodicBoundary(iNode->GetIndex())){
                // there is a second surface for this edge
                unsigned short mark; 
                for( auto i = 0u; i < config->GetnMarker_All(); i++){
                  if(geometry->nodes->GetVertex(geometry->vertex[iNode->GetMarker()][pointID]->GetNode(), i) != -1 && i != iNode->GetMarker()){
                    mark = i;
                    break;
                  }
                }

                auto jNode = geometry->nodes->GetVertex(geometry->vertex[iNode->GetMarker()][pointID]->GetNode(), mark);

                normal = geometry->vertex[mark][jNode]->GetNormal();
                
                dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

                norm_magnitude =  GeometryToolbox::Norm(nDim, normal);
                
                for(auto iDim = 0u; iDim < nDim; iDim++){
                  iNode->AddNewCoord(-dot_product*normal[iDim]/pow(norm_magnitude,2), iDim);
                }
              }

              su2double var_coord[nDim];

              for(auto iDim = 0u; iDim < nDim; iDim++){
                var_coord[iDim] = (iNode->GetNewCoord()[iDim]  - geometry->nodes->GetCoord(iNode->GetIndex())[iDim])*((su2double)config->GetGridDef_Nonlinear_Iter()); //TODO  var_coord can likely be obtained better
              }

              geometry->vertex[iNode->GetMarker()][iNode->GetVertex()]->SetVarCoord(var_coord); 

            }
          }
        
          ControlNodes.push_back(&red_SlideEdgeNodes);
          BdryNodes.push_back(&SlideEdgeNodes);
          Get_nCtrlNodes();
          types.push_back("EDGE");
          
          if(red_SlideSurfNodes.size() > 0){
            GetInterpCoeffs(geometry, config, type, radius);

            // construction ad tree
            
            /*--- Vector storing the coordinates of the boundary nodes ---*/
            vector<su2double> Coord_bound((red_SlideSurfNodes.size()+SlideSurfNodes.size()) * nDim);

            /*--- Vector storing the IDs of the boundary nodes ---*/
            vector<unsigned long> PointIDs((red_SlideSurfNodes.size()+SlideSurfNodes.size()));

            for ( auto iVertex = 0ul; iVertex < SlideSurfNodes.size(); iVertex++){

              // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
              PointIDs[iVertex] = SlideSurfNodes[iVertex]->GetVertex(); 
              for(auto iDim = 0u; iDim < nDim; iDim++){
                Coord_bound[ iVertex*nDim + iDim]  = geometry->nodes->GetCoord(SlideSurfNodes[iVertex]->GetIndex(), iDim);
              }
            }

            for ( auto iVertex = 0ul; iVertex < red_SlideSurfNodes.size(); iVertex++){

              // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
              PointIDs[iVertex + SlideSurfNodes.size()] = red_SlideSurfNodes[iVertex]->GetVertex();  
              for(auto iDim = 0u; iDim < nDim; iDim++){
                Coord_bound[ (SlideSurfNodes.size() + iVertex) * nDim + iDim]  = geometry->nodes->GetCoord(red_SlideSurfNodes[iVertex]->GetIndex(), iDim);
              }
            }
            
            /*--- Number of non-selected boundary nodes ---*/
            const unsigned long nVertexBound = PointIDs.size();

            /*--- Construction of AD tree ---*/
            CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);
            
            
            // free deformation
            
            GetFreeDeformation(geometry, config, type, radius, &red_SlideSurfNodes);
            
            /*--- ID of nearest boundary node ---*/
            unsigned long pointID;
            /*--- Distance to nearest boundary node ---*/
            su2double dist;
            /*--- rank of nearest boundary node ---*/
            int rankID;

            // loop through the sliding nodes 
            for(auto iNode : red_SlideSurfNodes){
              BoundADT.DetermineNearestNode(iNode->GetNewCoord(), dist, pointID, rankID);

              // geometry->vertex[SlideSurfNodes[]]
              // cout << iNode << " " << SlideSurfNodes[iNode]->GetIndex() << " Nearest node: " << SlideSurfNodes[pointID]->GetIndex() << endl;
              auto normal =  geometry->vertex[iNode->GetMarker()][pointID]->GetNormal();
              // cout << normal[0] << " " << normal[1] <<" " << normal[2] << endl;
              
              su2double dist_vec[nDim];
              GeometryToolbox::Distance(nDim, iNode->GetNewCoord(), geometry->nodes->GetCoord(geometry->vertex[iNode->GetMarker()][pointID]->GetNode()), dist_vec);

              auto dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

              auto norm_magnitude =  GeometryToolbox::Norm(nDim, normal);


              for(auto iDim = 0u; iDim < nDim; iDim++){
                iNode->AddNewCoord(-dot_product*normal[iDim]/pow(norm_magnitude,2), iDim);
              }

              su2double var_coord[nDim];

              for(auto iDim = 0u; iDim < nDim; iDim++){
                var_coord[iDim] = (iNode->GetNewCoord()[iDim]  - geometry->nodes->GetCoord(iNode->GetIndex())[iDim])*((su2double)config->GetGridDef_Nonlinear_Iter()); //TODO  var_coord can likely be obtained better
              }

              geometry->vertex[iNode->GetMarker()][iNode->GetVertex()]->SetVarCoord(var_coord); 

            }
          }

          ControlNodes.push_back(&red_SlideSurfNodes);
          BdryNodes.push_back(&SlideSurfNodes);
          Get_nCtrlNodes();
          types.push_back("SURF");

          GetInterpCoeffs(geometry, config, type, radius);
          GetInterpError(geometry, config, type, radius, maxErrorLocal, maxErrorNodes, maxErrorVector);
          SU2_MPI::Allreduce(&maxErrorLocal, &MaxErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());

          iter_surf++;
          
          if (rank == MASTER_NODE && iter_surf % 1 == 0){
            cout << "domain surf iter nr: " << iter_surf << ",\t Nr. of ctrl nodes: " << nCtrlNodesGlobal <<  ",\t error: " << MaxErrorGlobal << ",\t tol: " << DataReductionTolerance << endl;
          }
        }   

      }



      ofstream of;
      of.open("reducedslide.txt");
      for (auto x : red_SlideEdgeNodes){
          of << x->GetIndex() << endl;
        }
      of.close();   
  }else{
    
    /*--- First deforming the inflation layer ---*/
    if(config->GetRBF_IL_Preservation()){
            
      for(auto iLayer = 0u; iLayer < config->GetnMarker_Deform_Mesh_IL_Wall(); iLayer++){
        ControlNodes.resize(1);
        ControlNodes[0] = test[iLayer]; // set control nodes
        Get_nCtrlNodes();   // update number of control nodes

        /*--- Solve for interpolation coefficient using solely wall nodes as ctrl nodes ---*/
        GetInterpCoeffs(geometry, config, type, radius);

        /*--- Obtain the required deformation of the inflation layer edge ---*/
        GetIL_EdgeDeformation(geometry, config, type, radius, iLayer);

        /*--- Adding the inflation layer edge nodes to the control nodes ---*/
        ControlNodes.push_back(test_edge[iLayer]);
        Get_nCtrlNodes();

        /*--- Solve for interpolation coefficients ---*/
        GetInterpCoeffs(geometry, config, type, radius);

        /*--- Updating the inflation layer coordinates ---*/ 
        UpdateGridCoord(geometry, config, type, radius, *IL_internalNodes[iLayer], true);
      }
      
      /*--- Set control nodes for the domain outside inflation layer ---*/
      ControlNodes.resize(1);
      ControlNodes[0] = &BoundNodes;
      for(auto iLayer = 0u; iLayer < config->GetnMarker_Deform_Mesh_IL_Wall(); iLayer++){
        ControlNodes.push_back(test_edge[iLayer]);
      }
      Get_nCtrlNodes();
    }
    
    /*--- Obtaining the interpolation coefficients. ---*/
    GetInterpCoeffs(geometry, config, type, radius);

    // in case of sliding
    if (SlideEdgeNodes.size() != 0 ){
      
      // construction ad tree
      
      /*--- Vector storing the coordinates of the boundary nodes ---*/
      vector<su2double> Coord_bound(SlideEdgeNodes.size() * nDim);

      /*--- Vector storing the IDs of the boundary nodes ---*/
      vector<unsigned long> PointIDs(SlideEdgeNodes.size());

      for ( auto iVertex = 0ul; iVertex < SlideEdgeNodes.size(); iVertex++){

        // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
        PointIDs[iVertex] = SlideEdgeNodes[iVertex]->GetVertex(); 
        for(auto iDim = 0u; iDim < nDim; iDim++){
          Coord_bound[ iVertex*nDim + iDim]  = geometry->nodes->GetCoord(SlideEdgeNodes[iVertex]->GetIndex(), iDim);
        }
      }
      
      /*--- Number of non-selected boundary nodes ---*/
      const unsigned long nVertexBound = PointIDs.size();

      /*--- Construction of AD tree ---*/
      CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);
      
      
      // free deformation
      
      GetFreeDeformation(geometry, config, type, radius, &SlideEdgeNodes);
      
       /*--- ID of nearest boundary node ---*/
      unsigned long pointID;
      /*--- Distance to nearest boundary node ---*/
      su2double dist;
      /*--- rank of nearest boundary node ---*/
      int rankID;

      // loop through the sliding nodes 
      for(auto iNode : SlideEdgeNodes){
        BoundADT.DetermineNearestNode(iNode->GetNewCoord(), dist, pointID, rankID);

        auto normal =  geometry->vertex[iNode->GetMarker()][pointID]->GetNormal();
        
        su2double dist_vec[nDim];
        GeometryToolbox::Distance(nDim, iNode->GetNewCoord(), geometry->nodes->GetCoord(geometry->vertex[iNode->GetMarker()][pointID]->GetNode()), dist_vec);

        auto dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

        auto norm_magnitude =  GeometryToolbox::Norm(nDim, normal);
        
        for(auto iDim = 0u; iDim < nDim; iDim++){ 
          iNode->AddNewCoord(-dot_product*normal[iDim]/pow(norm_magnitude,2), iDim);
        }

        if(nDim == 3 &&  !geometry->nodes->GetPeriodicBoundary(iNode->GetIndex())){
          // there is a second surface for this edge
          unsigned short mark; 
          for( auto i = 0u; i < config->GetnMarker_All(); i++){
            if(geometry->nodes->GetVertex(geometry->vertex[iNode->GetMarker()][pointID]->GetNode(), i) != -1 && i != iNode->GetMarker()){
              mark = i;
              break;
            }
          }

          auto jNode = geometry->nodes->GetVertex(geometry->vertex[iNode->GetMarker()][pointID]->GetNode(), mark);

          normal = geometry->vertex[mark][jNode]->GetNormal();
          
          dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

          norm_magnitude =  GeometryToolbox::Norm(nDim, normal);
          
          for(auto iDim = 0u; iDim < nDim; iDim++){
            iNode->AddNewCoord(-dot_product*normal[iDim]/pow(norm_magnitude,2), iDim);
          }
        }

        
        su2double var_coord[nDim];
        for(auto iDim = 0u; iDim < nDim; iDim++){
          var_coord[iDim] = (iNode->GetNewCoord()[iDim]  - geometry->nodes->GetCoord(iNode->GetIndex())[iDim])*((su2double)config->GetGridDef_Nonlinear_Iter()); //TODO  var_coord can likely be obtained better
        }

        geometry->vertex[iNode->GetMarker()][iNode->GetVertex()]->SetVarCoord(var_coord); 

      }

      
      ControlNodes.push_back(&SlideEdgeNodes);
      Get_nCtrlNodes();
      GetInterpCoeffs(geometry, config, type, radius);
    }

    // SLIDING SURFACE NODES!!
    if (SlideSurfNodes.size() != 0 ){
    
      // construction ad tree
      
      /*--- Vector storing the coordinates of the boundary nodes ---*/
      vector<su2double> Coord_bound(SlideSurfNodes.size() * nDim);

      /*--- Vector storing the IDs of the boundary nodes ---*/
      vector<unsigned long> PointIDs(SlideSurfNodes.size());

      for ( auto iVertex = 0ul; iVertex < SlideSurfNodes.size(); iVertex++){

        // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
        PointIDs[iVertex] = SlideSurfNodes[iVertex]->GetVertex(); 
        for(auto iDim = 0u; iDim < nDim; iDim++){
          Coord_bound[ iVertex*nDim + iDim]  = geometry->nodes->GetCoord(SlideSurfNodes[iVertex]->GetIndex(), iDim);
        }
      }
      
      /*--- Number of non-selected boundary nodes ---*/
      const unsigned long nVertexBound = PointIDs.size();

      /*--- Construction of AD tree ---*/
      CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);
      
      
      // free deformation
      
      GetFreeDeformation(geometry, config, type, radius, &SlideSurfNodes);
      
        /*--- ID of nearest boundary node ---*/
      unsigned long pointID;
      /*--- Distance to nearest boundary node ---*/
      su2double dist;
      /*--- rank of nearest boundary node ---*/
      int rankID;

      // loop through the sliding nodes 
      for(auto iNode : SlideSurfNodes){
        BoundADT.DetermineNearestNode(iNode->GetNewCoord(), dist, pointID, rankID);

        auto normal =  geometry->vertex[iNode->GetMarker()][pointID]->GetNormal();
        
        su2double dist_vec[nDim];
        GeometryToolbox::Distance(nDim, iNode->GetNewCoord(), geometry->nodes->GetCoord(geometry->vertex[iNode->GetMarker()][pointID]->GetNode()), dist_vec);

        auto dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

        auto norm_magnitude =  GeometryToolbox::Norm(nDim, normal);
        
        for(auto iDim = 0u; iDim < nDim; iDim++){
          iNode->AddNewCoord(-dot_product*normal[iDim]/pow(norm_magnitude,2), iDim);
        }

        
        su2double var_coord[nDim];
        for(auto iDim = 0u; iDim < nDim; iDim++){
          var_coord[iDim] = (iNode->GetNewCoord()[iDim]  - geometry->nodes->GetCoord(iNode->GetIndex())[iDim])*((su2double)config->GetGridDef_Nonlinear_Iter()); //TODO  var_coord can likely be obtained better
        }

        geometry->vertex[iNode->GetMarker()][iNode->GetVertex()]->SetVarCoord(var_coord); 

      }
      
      ControlNodes.push_back(&SlideSurfNodes);
      Get_nCtrlNodes();
      GetInterpCoeffs(geometry, config, type, radius);
    }
  }
}

void CRadialBasisFunctionInterpolation::GetInterpCoeffs(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius){
  
  /*--- Obtaining the control nodes coordinates and distributing over all processes. ---*/
  SetCtrlNodeCoords(geometry);

  /*--- Obtaining the deformation of the control nodes. ---*/
  SetDeformation(geometry, config);

  /*--- Computation of the (inverse) interpolation matrix. ---*/
  su2passivematrix invInterpMat;
  ComputeInterpolationMatrix(geometry, config, type, radius, invInterpMat);

  /*--- Obtaining the interpolation coefficients. ---*/
  ComputeInterpCoeffs(invInterpMat);
}


void CRadialBasisFunctionInterpolation::SetBoundNodes(CGeometry* geometry, CConfig* config){
  
  /*--- Storing of the local node, marker and vertex information of the boundary nodes ---*/

  /*--- Looping over the markers ---*/
  for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++) {

    // if (config->GetMarker_All_PerBound(iMarker)){ cout << iMarker << " " <<  config->GetMarker_All_TagBound(iMarker) << endl;
    // cout << config->GetMarker_Periodic_Donor(config->GetMarker_All_TagBound(iMarker)) << endl;
    // }

    /*--- Checking if not internal or send/receive marker ---*/
    if (!config->GetMarker_All_Deform_Mesh_Internal(iMarker) && !config->GetMarker_All_SendRecv(iMarker) && !config->GetMarker_All_Deform_Mesh_IL_Wall(iMarker) && !config->GetMarker_All_Deform_Mesh_Slide(iMarker) && !config->GetMarker_All_PerBound(iMarker)) {

      /*--- Looping over the vertices of marker ---*/
      for (auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++) {
        /*--- Node in consideration ---*/
        auto iNode = geometry->vertex[iMarker][iVertex]->GetNode();
        if(geometry->nodes->GetPeriodicBoundary(iNode)){

          // find the associated periodic marker. 

          // find the donor periodic markers

          // if the associated periodic marker is the larger one of the two then disregard it.
          // if it is the smaller of the two then it is added as control node. 
          unsigned short perMarker;
          for (auto jMarker = 0u; jMarker < config->GetnMarker_All(); jMarker++){
            if(geometry->nodes->GetVertex(iNode, jMarker) != -1 && config->GetMarker_All_PerBound(jMarker)){
              perMarker = jMarker;
              if(perMarker < config->GetMarker_Periodic_Donor(config->GetMarker_All_TagBound(jMarker))){
                BoundNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
              }
              break;
            }
          }



        }else{

          

          /*--- Check whether node is part of the subdomain and not shared with a receiving marker (for parallel computation)*/
          if (geometry->nodes->GetDomain(iNode)) {
            BoundNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));        
          }        
        }
      }
    }

    if(config->GetMarker_All_Deform_Mesh_IL_Wall(iMarker)){
      test.push_back(new vector<CRadialBasisFunctionNode*>);
      for ( auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++){
        auto iNode =  geometry->vertex[iMarker][iVertex]->GetNode();
        IL_WallNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
        test.back()->push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
      }
    }
  }

  // This is done seperately for the sliding nodes, as the boundNodes requires the right iMarker when adding a CRadialBasisFunctionNode
  for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++){
    if (config->GetMarker_All_Deform_Mesh_Slide(iMarker)){
      for (auto iVertex = 0u; iVertex < geometry->nVertex[iMarker]; iVertex++){
        auto iNode = geometry->vertex[iMarker][iVertex]->GetNode();

        unsigned short nVertex = 0;
        for (auto jMarker = 0u; jMarker < config->GetnMarker_All(); jMarker++ ){
          if (geometry->nodes->GetVertex(iNode, jMarker) != -1){
            if (!config->GetMarker_All_SendRecv(jMarker)){
              nVertex++;
            }
          }
        }
        
        if(nDim == 2){
          if (nVertex > 1){
            BoundNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
          }
          else{ // TODO is it still possible to be a among BoundNodes??
            // cout << iNode << endl;
            SlideEdgeNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
          }
        }
        //ndim == 3
        else{
          if (nVertex == 2){ 
            if(find_if (BoundNodes.begin(), BoundNodes.end(), [&](CRadialBasisFunctionNode* i){return i->GetIndex() == iNode;}) == BoundNodes.end()){
              if(geometry->nodes->GetPeriodicBoundary(iNode)){

                // find the associated periodic marker. 

                // find the donor periodic markers

                // if the associated periodic marker is the larger one of the two then disregard it.
                // if it is the smaller of the two then it is added as control node. 
                unsigned short perMarker;
                for (auto jMarker = 0u; jMarker < config->GetnMarker_All(); jMarker++){
                  if(geometry->nodes->GetVertex(iNode, jMarker) != -1 && config->GetMarker_All_PerBound(jMarker)){
                    perMarker = jMarker;
                    if(perMarker < config->GetMarker_Periodic_Donor(config->GetMarker_All_TagBound(jMarker))){
                      SlideEdgeNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
                    }
                    break;
                  }
                }



              }else{
                SlideEdgeNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));               
              }
            }
          }
          else if(nVertex == 1) {
            if(find_if (BoundNodes.begin(), BoundNodes.end(), [&](CRadialBasisFunctionNode* i){return i->GetIndex() == iNode;}) == BoundNodes.end()){
              SlideSurfNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex)); 
            } 
          }
          else{ 
            if(geometry->nodes->GetPeriodicBoundary(iNode)){

                // find the associated periodic marker. 

                // find the donor periodic markers

                // if the associated periodic marker is the larger one of the two then disregard it.
                // if it is the smaller of the two then it is added as control node. 
                unsigned short perMarker;
                for (auto jMarker = 0u; jMarker < config->GetnMarker_All(); jMarker++){
                  if(geometry->nodes->GetVertex(iNode, jMarker) != -1 && config->GetMarker_All_PerBound(jMarker)){
                    perMarker = jMarker;
                    if(perMarker < config->GetMarker_Periodic_Donor(config->GetMarker_All_TagBound(jMarker))){
                      BoundNodes.push_back(new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
                    }
                    break;
                  }
                }



              }else{
              BoundNodes.push_back( new CRadialBasisFunctionNode(iNode, iMarker, iVertex));
              }
          }
        }

      }
    }
  }

  /*--- Sorting of the boundary nodes based on their index ---*/
  stable_sort(BoundNodes.begin(), BoundNodes.end(), HasSmallerIndex);
  stable_sort(SlideEdgeNodes.begin(), SlideEdgeNodes.end(), HasSmallerIndex);

  /*--- Obtaining unique set ---*/
  BoundNodes.resize(distance(BoundNodes.begin(), unique(BoundNodes.begin(), BoundNodes.end(), HasEqualIndex)));
  SlideEdgeNodes.resize(distance(SlideEdgeNodes.begin(), unique(SlideEdgeNodes.begin(), SlideEdgeNodes.end(), HasEqualIndex)));

  ofstream of;
  of.open("boundnodes.txt");
  for(auto x : BoundNodes){ of << x->GetIndex() << endl;}
  of.close();

  of.open("wallnodes.txt");
  // for(auto x : IL_WallNodes){ of << x->GetIndex() << endl;}
  for (auto x : test){
    for (auto y : *x){
      of << y->GetIndex() << endl;
    }
  }
  of.close();

  of.open("slideEdgeNodes.txt");
  for(auto x : SlideEdgeNodes){ of << x->GetIndex() << endl;}
  of.close();

  of.open("slideSurfNodes.txt");
  for(auto x : SlideSurfNodes){ of << x->GetIndex() << endl;}
  of.close();
  cout << "written node files! "<< endl;
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

void CRadialBasisFunctionInterpolation::ComputeInterpolationMatrix(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, su2passivematrix& invInterpMat){

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
        // auto dist = GetDistance(geometry, config, nDim, CtrlCoords[iNode*nDim], CtrlCoords[jNode*nDim]);
        // su2double dist;
        // if(config->GetnMarker_Periodic() > 0){
        //   dist = GetDistance(geometry, config, nDim, CtrlCoords[iNode*nDim], CtrlCoords[jNode*nDim]);
        // }else{
        //   dist = GeometryToolbox::Distance(nDim, CtrlCoords[iNode*nDim], CtrlCoords[jNode*nDim]);   
        // }
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
    su2double IL_height = config->GetRBF_IL_Height();

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
      test_edge.resize(config->GetnMarker_Deform_Mesh_IL_Wall());
      for( auto iMarker = 0u; iMarker < config->GetnMarker_Deform_Mesh_IL_Wall(); iMarker++){
        IL_internalNodes[iMarker] = new vector<unsigned long>;
        test_edge[iMarker] = new vector<CRadialBasisFunctionNode*>;
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
        if(abs(dist-IL_height) < 1e-6){
          // cout << geometry->nodes->GetCoord(iNode)[0] << "\t" << geometry->nodes->GetCoord(iNode)[1] << "\t"<<  dist << "\t" << nearest_marker << "\t" << nearest_elem << endl;
          IL_EdgeNodes.push_back(new CRadialBasisFunctionNode(iNode, nearest_marker, nearest_elem));
          test_edge[nearest_marker]->push_back(new CRadialBasisFunctionNode(iNode, nearest_marker, nearest_elem));
        }else if(dist-IL_height < 0){
          IL_internalNodes[nearest_marker]->push_back(iNode);
        }else{
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

          /*--- if not among the boundary and wall nodes ---*/  //TODO check how this works for multiple ILs
          if (find_if (BoundNodes.begin(), BoundNodes.end(), [&](CRadialBasisFunctionNode* i){return i->GetIndex() == iNode;}) == BoundNodes.end() &&
                find_if (IL_WallNodes.begin(), IL_WallNodes.end(), [&](CRadialBasisFunctionNode* i){return i->GetIndex() == iNode;}) == IL_WallNodes.end()) {
            ElemADT.DetermineNearestElement(geometry->nodes->GetCoord(iNode), dist, nearest_marker, nearest_elem, rankID);

            if(abs(dist-IL_height) < 1e-6){ //TODO what if inflation layers have different heights?
              // cout << geometry->nodes->GetCoord(iNode)[0] << "\t" << geometry->nodes->GetCoord(iNode)[1] << "\t"<<  dist << "\t" << nearest_marker << "\t" << nearest_elem << endl;
              IL_EdgeNodes.push_back(new CRadialBasisFunctionNode(iNode, nearest_marker, nearest_elem));
            }else if(dist-IL_height < 0){
              IL_internalNodes[nearest_marker]->push_back(iNode);
            }else{
              internalNodes.push_back(iNode);
            }
          }
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
    // for(auto x : IL_EdgeNodes){
    //   ofile << x->GetIndex() << endl;
    // }
    for (auto x : test_edge){
      for (auto y : *x){
        ofile << y->GetIndex() << endl;
      }
    }
    ofile.close();

    ofile.open("il_internal_nodes.txt");
    // for(auto x : IL_internalNodes){
    for( auto iLayer = 0u; iLayer < config->GetnMarker_Deform_Mesh_IL_Wall(); iLayer++){
      for(auto y : *IL_internalNodes[iLayer]){ ofile << y << endl;}
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
    
    /*--- Adding nodes on markers considered as internal nodes ---*/
    for (auto iMarker = 0u; iMarker < geometry->GetnMarker(); iMarker++){

      /*--- Check if marker is considered as internal nodes ---*/
      if(config->GetMarker_All_Deform_Mesh_Internal(iMarker) || config->GetMarker_All_PerBound(iMarker)){
        
        /*--- Loop over marker vertices ---*/
        for (auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++) {

          /*--- Local node index ---*/
          auto iNode = geometry->vertex[iMarker][iVertex]->GetNode();

          /*--- if not among the boundary nodes ---*/
          if (find_if (BoundNodes.begin(), BoundNodes.end(), [&](CRadialBasisFunctionNode* i){return i->GetIndex() == iNode;}) == BoundNodes.end() && 
                  find_if (SlideEdgeNodes.begin(), SlideEdgeNodes.end(), [&](CRadialBasisFunctionNode* i){return i->GetIndex() == iNode;}) == SlideEdgeNodes.end()) {
            internalNodes.push_back(iNode);
          }
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
    internalNodes.resize(distance(internalNodes.begin(), unique(internalNodes.begin(), internalNodes.end())));
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

void CRadialBasisFunctionInterpolation::UpdateGridCoord(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, const vector<unsigned long>& internalNodes, bool inflationLayer){

  if(rank == MASTER_NODE){
    cout << "updating the grid coordinates" << endl;
  }

  /*--- Update of internal node coordinates ---*/

  UpdateInternalCoords(geometry, config, type, radius, internalNodes);
  
  /*--- Update of boundary node coordinates ---*/
  UpdateBoundCoords(geometry, config, type, radius, inflationLayer);   


  /*--- In case of data reduction, perform the correction for nonzero error nodes ---*/
  if(config->GetRBF_DataReduction() && (BoundNodes.size() > 0 || SlideEdgeNodes.size()> 0 )){
    SetCorrection(geometry, config, type, internalNodes, inflationLayer);
  }
  
}

void CRadialBasisFunctionInterpolation::UpdateInternalCoords(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, const vector<unsigned long>& internalNodes){
  
   /*--- Vector for storing the coordinate variation ---*/
  su2double var_coord[nDim]{0.0};
  
  /*--- Loop over the internal nodes ---*/
  for(auto iNode = 0ul; iNode < internalNodes.size(); iNode++){

    /*--- Loop for contribution of each control node ---*/
    for(auto jNode = 0ul; jNode < nCtrlNodesGlobal; jNode++){

      /*--- Determine distance between considered internal and control node ---*/
      // su2double dist;
      // if(config->GetnMarker_Periodic() > 0){
      //   dist = GetDistance(geometry, config, nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(internalNodes[iNode]));
      // }else{
      //   dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(internalNodes[iNode]));
      // }
      auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(internalNodes[iNode]));
      // auto dist = GetDistance(geometry, config, nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(internalNodes[iNode]));

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

void CRadialBasisFunctionInterpolation::UpdateBoundCoords(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, bool inflationLayer){
  
  /*--- Vector for storing the coordinate variation ---*/
  su2double var_coord[nDim]{0.0};
  
  /*--- In case of data reduction, the non-control boundary nodes are treated as if they where internal nodes ---*/
  if(config->GetRBF_DataReduction()){

    for(auto iNodes : BdryNodes){
      for(auto iNode : *iNodes){
        for( auto jNode = 0ul; jNode < nCtrlNodesGlobal; jNode++){
          /*--- Distance of non-selected boundary node to control node ---*/
          auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
          // auto dist = GetDistance(geometry, config, nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(iNode->GetIndex()));

          // su2double dist;
          // if(config->GetnMarker_Periodic() > 0){
          //   dist = GetDistance(geometry, config, nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
          // }else{
          //   dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
          // }
          /*--- Evaluation of the radial basis function based on the distance ---*/
          auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));

          /*--- Computing and add the resulting coordinate variation ---*/
          for(auto iDim = 0u; iDim < nDim; iDim++){
            var_coord[iDim] += rbf*InterpCoeff[jNode * nDim + iDim];
          }
        }
      
        /*--- Applying the coordinate variation and resetting the var_coord vector*/
        for(auto iDim = 0u; iDim < nDim; iDim++){
          geometry->nodes->AddCoord(iNode->GetIndex(), iDim, var_coord[iDim]);
          var_coord[iDim] = 0;
        }
      }
      if(inflationLayer){break;}
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
    if(inflationLayer){break;}
  } 
}


void CRadialBasisFunctionInterpolation::GetInitMaxErrorNode(CGeometry* geometry, CConfig* config, vector<CRadialBasisFunctionNode*>& movingNodes,  vector<unsigned long>& maxErrorNodes, vector<unsigned short>& maxErrorVector, su2double& maxErrorLocal){

  /*--- Set max error to zero ---*/
  maxErrorLocal = 0.0;

  unsigned long maxErrorNodeLocal;
  unsigned short maxErrorVectorLocal; 

  for (auto iNodes = 0u; iNodes < BdryNodes.size(); iNodes++ ){
    /*--- Loop over the nodes ---*/  
    for(auto iNode = 0ul; iNode < BdryNodes[iNodes]->size(); iNode++){ 
      
      su2double* varCoord = geometry->vertex[(*BdryNodes[iNodes])[iNode]->GetMarker()][(*BdryNodes[iNodes])[iNode]->GetVertex()]->GetVarCoord();
      (*BdryNodes[iNodes])[iNode]->SetError(varCoord,nDim);

      /*--- Compute to squared norm of the deformation ---*/
      su2double normSquaredDeformation = GeometryToolbox::SquaredNorm(nDim, varCoord);    

      /*--- In case squared norm deformation is larger than the error, update the error ---*/
      if(normSquaredDeformation > maxErrorLocal){
        maxErrorLocal = normSquaredDeformation;
        maxErrorNodeLocal = iNode;
        maxErrorVectorLocal = iNodes;
      }
    }
  }
  
  if(maxErrorLocal > 0){
    maxErrorNodes.push_back(maxErrorNodeLocal);
    maxErrorVector.push_back(maxErrorVectorLocal);
    /*--- Account for the possibility of applying the deformation in multiple steps ---*/
    maxErrorLocal = sqrt(maxErrorLocal) / ((su2double)config->GetGridDef_Nonlinear_Iter());

    /*--- Steps for finding a node on a secondary edge (double edged greedy algorithm) ---*/

    /*--- Total error, defined as total variation in coordinates ---*/
    su2double* errorTotal = geometry->vertex[movingNodes[maxErrorNodeLocal]->GetMarker()][movingNodes[maxErrorNodeLocal]->GetVertex()]->GetVarCoord();
    
    /*--- Making a copy of the total error to errorStep, to allow for manipulation of the data ---*/
    su2double* errorStep = new su2double[nDim];
    copy(errorTotal, errorTotal+nDim, errorStep);

    /*--- Account for applying deformation in multiple steps ---*/
    for( unsigned short iDim = 0u; iDim < nDim; iDim++){
      errorStep[iDim] = errorStep[iDim]/(su2double)config->GetGridDef_Nonlinear_Iter();
    }
    
    /*--- Finding a double edged error node if possible ---*/
    GetDoubleEdgeNode(errorStep, maxErrorNodes, maxErrorVector);
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


void CRadialBasisFunctionInterpolation::GetInterpError(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, su2double& maxErrorLocal, vector<unsigned long>& maxErrorNodes, vector <unsigned short>& maxErrorVector){
  
  /*--- Array containing the local error ---*/
  su2double localError[nDim];

  /*--- Magnitude of the local maximum error ---*/
  maxErrorLocal = 0.0;
  unsigned long maxErrorNodeLocal;
  unsigned short maxErrorVectorLocal;



  // unsigned short vec_cnt = 0;
  for ( auto iNodes = 0u; iNodes < BdryNodes.size(); iNodes++ ){
    
    
    // if(types[iNodes] == "SURF"){
    //     // free deformation  
    //   GetFreeDeformation(geometry, config, type, radius, &SlideSurfNodes);

    //   /*--- Vector storing the coordinates of the boundary nodes ---*/
    //   vector<su2double> Coord_bound((red_SlideSurfNodes.size()+SlideSurfNodes.size()) * nDim);

    //   /*--- Vector storing the IDs of the boundary nodes ---*/
    //   vector<unsigned long> PointIDs((red_SlideSurfNodes.size()+SlideSurfNodes.size()));

    //   for ( auto iVertex = 0ul; iVertex < SlideSurfNodes.size(); iVertex++){

    //     // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
    //     PointIDs[iVertex] = SlideSurfNodes[iVertex]->GetVertex(); 
    //     for(auto iDim = 0u; iDim < nDim; iDim++){
    //       Coord_bound[ iVertex*nDim + iDim]  = geometry->nodes->GetCoord(SlideSurfNodes[iVertex]->GetIndex(), iDim);
    //     }
    //   }

    //   for ( auto iVertex = 0ul; iVertex < red_SlideSurfNodes.size(); iVertex++){

    //     // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
    //     PointIDs[iVertex + SlideSurfNodes.size()] = red_SlideSurfNodes[iVertex]->GetVertex();  
    //     for(auto iDim = 0u; iDim < nDim; iDim++){
    //       Coord_bound[ (SlideSurfNodes.size() + iVertex) * nDim + iDim]  = geometry->nodes->GetCoord(red_SlideSurfNodes[iVertex]->GetIndex(), iDim);
    //     }
    //   }
      
    //   /*--- Number of non-selected boundary nodes ---*/
    //   const unsigned long nVertexBound = PointIDs.size();

    //   /*--- Construction of AD tree ---*/
    //   CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);
      
      
    
      
    //   /*--- ID of nearest boundary node ---*/
    //   unsigned long pointID;
    //   /*--- Distance to nearest boundary node ---*/
    //   su2double dist;
    //   /*--- rank of nearest boundary node ---*/
    //   int rankID;

    // // // loop through the sliding nodes 
    // // for(auto iNode : SlideEdgeNodes){
    //  for ( auto iNode = 0ul; iNode < BdryNodes[iNodes]->size(); iNode++){
    //     BoundADT.DetermineNearestNode((*BdryNodes[iNodes])[iNode]->GetNewCoord(), dist, pointID, rankID);

    //     // geometry->vertex[SlideSurfNodes[]]
    //     // cout << iNode << " " << SlideSurfNodes[iNode]->GetIndex() << " Nearest node: " << SlideSurfNodes[pointID]->GetIndex() << endl;
    //     auto normal =  geometry->vertex[(*BdryNodes[iNodes])[iNode]->GetMarker()][pointID]->GetNormal();
    //     // cout << normal[0] << " " << normal[1] <<" " << normal[2] << endl;
        
    //     su2double dist_vec[nDim];
    //     GeometryToolbox::Distance(nDim, (*BdryNodes[iNodes])[iNode]->GetNewCoord(), geometry->nodes->GetCoord(geometry->vertex[(*BdryNodes[iNodes])[iNode]->GetMarker()][pointID]->GetNode()), dist_vec);

    //     auto dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);
    //     auto norm_magnitude =  GeometryToolbox::Norm(nDim, normal);

    //     for (auto iDim = 0u; iDim < nDim; iDim++ ){
    //       localError[iDim] = dot_product*normal[iDim]/pow(norm_magnitude,2);
    //       // localError[iDim] = 0;
    //     }
    //     (*BdryNodes[iNodes])[iNode]->SetError(localError,nDim);

    //     /*--- Compute error magnitude and update local maximum error if necessary ---*/
    //     su2double errorMagnitude = GeometryToolbox::Norm(nDim, localError);
    //     if(errorMagnitude > maxErrorLocal){
    //       maxErrorLocal = errorMagnitude;
    //       maxErrorNodeLocal = iNode;
    //       maxErrorVectorLocal = iNodes;

    //     }
    //  }

    // }else if (types[iNodes] == "EDGE"){
    //   GetFreeDeformation(geometry, config, type, radius, &SlideEdgeNodes);

    //    // construction ad tree
          
    //     /*--- Vector storing the coordinates of the boundary nodes ---*/
    //     vector<su2double> Coord_bound((red_SlideEdgeNodes.size()+SlideEdgeNodes.size()) * nDim);

    //     /*--- Vector storing the IDs of the boundary nodes ---*/
    //     vector<unsigned long> PointIDs((red_SlideEdgeNodes.size()+SlideEdgeNodes.size()));

    //     for ( auto iVertex = 0ul; iVertex < SlideEdgeNodes.size(); iVertex++){

    //       // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
    //       PointIDs[iVertex] = SlideEdgeNodes[iVertex]->GetVertex(); 
    //       for(auto iDim = 0u; iDim < nDim; iDim++){
    //         Coord_bound[ iVertex*nDim + iDim]  = geometry->nodes->GetCoord(SlideEdgeNodes[iVertex]->GetIndex(), iDim);
    //       }
    //     }

    //     for ( auto iVertex = 0ul; iVertex < red_SlideEdgeNodes.size(); iVertex++){

    //       // auto iNode = SlideEdgeNodes[iVertex]->GetVertex();
    //       PointIDs[iVertex + SlideEdgeNodes.size()] = red_SlideEdgeNodes[iVertex]->GetVertex();  
    //       for(auto iDim = 0u; iDim < nDim; iDim++){
    //         Coord_bound[ (SlideEdgeNodes.size() + iVertex) * nDim + iDim]  = geometry->nodes->GetCoord(red_SlideEdgeNodes[iVertex]->GetIndex(), iDim);
    //       }
    //     }
        
    //     /*--- Number of non-selected boundary nodes ---*/
    //     const unsigned long nVertexBound = PointIDs.size();

    //     /*--- Construction of AD tree ---*/
    //     CADTPointsOnlyClass BoundADT(nDim, nVertexBound, Coord_bound.data(), PointIDs.data(), false);
                
    //     /*--- ID of nearest boundary node ---*/
    //     unsigned long pointID;
    //     /*--- Distance to nearest boundary node ---*/
    //     su2double dist;
    //     /*--- rank of nearest boundary node ---*/
    //     int rankID;

    //     // // loop through the sliding nodes 
    //     // for(auto iNode : SlideEdgeNodes){
    //     for ( auto iNode = 0ul; iNode < BdryNodes[iNodes]->size(); iNode++){
    //       BoundADT.DetermineNearestNode((*BdryNodes[iNodes])[iNode]->GetNewCoord(), dist, pointID, rankID);

    //       // geometry->vertex[SlideSurfNodes[]]
    //       // cout << iNode << " " << SlideSurfNodes[iNode]->GetIndex() << " Nearest node: " << SlideSurfNodes[pointID]->GetIndex() << endl;
    //       auto normal =  geometry->vertex[(*BdryNodes[iNodes])[iNode]->GetMarker()][pointID]->GetNormal();
    //       // cout << normal[0] << " " << normal[1] <<" " << normal[2] << endl;
          
    //       su2double dist_vec[nDim];
    //       GeometryToolbox::Distance(nDim, (*BdryNodes[iNodes])[iNode]->GetNewCoord(), geometry->nodes->GetCoord(geometry->vertex[(*BdryNodes[iNodes])[iNode]->GetMarker()][pointID]->GetNode()), dist_vec);

    //       auto dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);
    //       auto norm_magnitude =  GeometryToolbox::Norm(nDim, normal);

    //       for (auto iDim = 0u; iDim < nDim; iDim++ ){
    //         localError[iDim] = dot_product*normal[iDim]/pow(norm_magnitude,2);
    //       }

    //       if(nDim == 3 && !geometry->nodes->GetPeriodicBoundary(geometry->vertex[(*BdryNodes[iNodes])[iNode]->GetMarker()][pointID]->GetNode())){
    //         // there is a second surface for this edge
    //         unsigned short mark; 
    //         for( auto i = 0u; i < config->GetnMarker_All(); i++){
    //           if(geometry->nodes->GetVertex(geometry->vertex[(*BdryNodes[iNodes])[iNode]->GetMarker()][pointID]->GetNode(), i) != -1 && i != (*BdryNodes[iNodes])[iNode]->GetMarker()){
    //             mark = i;
    //             break;
    //           }
    //         }
        
    //         auto jNode = geometry->nodes->GetVertex(geometry->vertex[(*BdryNodes[iNodes])[iNode]->GetMarker()][pointID]->GetNode(), mark);
    //         if(jNode != -1){
    //           normal = geometry->vertex[mark][jNode]->GetNormal();
              
    //           dot_product = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

    //           norm_magnitude =  GeometryToolbox::Norm(nDim, normal);
              
    //           for(auto iDim = 0u; iDim < nDim; iDim++){
    //             localError[iDim] += dot_product*normal[iDim]/pow(norm_magnitude,2);
    //             // iNode->AddNewCoord(-dot_product*normal[iDim]/pow(norm_magnitude,2), iDim);
    //           }
    //         }
    //       }
    //       (*BdryNodes[iNodes])[iNode]->SetError(localError,nDim);

    //       /*--- Compute error magnitude and update local maximum error if necessary ---*/
    //       su2double errorMagnitude = GeometryToolbox::Norm(nDim, localError);
    //       if(errorMagnitude > maxErrorLocal){
    //         maxErrorLocal = errorMagnitude;
    //         maxErrorNodeLocal = iNode;
    //         maxErrorVectorLocal = iNodes;

    //       }
    //     }
    // }
    // else if( types[iNodes] == "MOVING"){
      for ( auto iNode = 0ul; iNode < BdryNodes[iNodes]->size(); iNode++){

      
        GetNodalError(geometry, config, type, radius, (*BdryNodes[iNodes])[iNode], localError);
      
        (*BdryNodes[iNodes])[iNode]->SetError(localError,nDim);

        /*--- Compute error magnitude and update local maximum error if necessary ---*/
        su2double errorMagnitude = GeometryToolbox::Norm(nDim, localError);
        if(errorMagnitude > maxErrorLocal){
          maxErrorLocal = errorMagnitude;
          maxErrorNodeLocal = iNode;
          maxErrorVectorLocal = iNodes;

        }
      }
    // }
  }
  
  // for(auto iNodes = 0u; iNodes < BdryNodes.size(); iNodes++ ){
  //   unsigned long cnt = 0;
  //   for(auto iNode : *iNodes){
  //     // cout << iNode->GetIndex() << endl;
  //     /*--- Compute nodal error ---*/
      // GetNodalError(geometry, config, type, radius, iNode, localError);
      
  //     /*--- Setting error ---*/
  //     iNode->SetError(localError, nDim);

  //     /*--- Compute error magnitude and update local maximum error if necessary ---*/
  //     su2double errorMagnitude = GeometryToolbox::Norm(nDim, localError);
  //     if(errorMagnitude > maxErrorLocal){
  //       maxErrorLocal = errorMagnitude;
  //       maxErrorNodeLocal = cnt;
  //       // vec_idx = vec_cnt;
  //       cout << "updated max error: "  << cnt << " " << iNode->GetIndex() << endl;

  //     }
  //     cnt++;
  //   }
  //   // vec_cnt++;
  // }

  // /*--- Loop over non-selected boundary nodes ---*/
  // for(auto iNode = 0ul; iNode < BoundNodes.size(); iNode++){

  //   /*--- Compute nodal error ---*/
  //   // GetNodalError(geometry, config, type, radius, iNode, localError);

  //   /*--- Setting error ---*/
  //   BoundNodes[iNode]->SetError(localError, nDim);

  //   /*--- Compute error magnitude and update local maximum error if necessary ---*/
  //   su2double errorMagnitude = GeometryToolbox::Norm(nDim, localError);
  //   if(errorMagnitude > maxErrorLocal){
  //     maxErrorLocal = errorMagnitude;
  //     maxErrorNodeLocal = iNode;
  //   }
  // }  

  if(maxErrorLocal > 0){
    /*--- Including the maximum error nodes in the max error nodes vector ---*/
    maxErrorNodes.push_back(maxErrorNodeLocal);
    maxErrorVector.push_back(maxErrorVectorLocal);

    /*--- Finding a double edged error node if possible ---*/
    GetDoubleEdgeNode( (*BdryNodes[maxErrorVector[0]])[maxErrorNodes[0]]->GetError(), maxErrorNodes, maxErrorVector);
    
  }
}

void CRadialBasisFunctionInterpolation::GetNodalError(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, CRadialBasisFunctionNode* iNode, su2double* localError){ 
  
  /*--- If requested (no by default) impose the surface deflections in increments ---*/
  const su2double VarIncrement = 1.0 / ((su2double)config->GetGridDef_Nonlinear_Iter());
  
  /*--- If node is part of a moving boundary then the error is defined as the difference
           between the found and prescribed displacements. Thus, here the displacement is substracted from the error ---*/

    su2double* displacement;
    if(geometry->nodes->GetBoundary(iNode->GetIndex())){
      displacement = geometry->vertex[iNode->GetMarker()][iNode->GetVertex()]->GetVarCoord();
    }else{
      displacement = iNode->GetVarCoord();
    }

    for(auto iDim = 0u; iDim < nDim; iDim++){
      localError[iDim] = -displacement[iDim] * VarIncrement;
    }

  /*--- Resulting displacement from the RBF interpolation is added to the error ---*/

  /*--- Finding contribution of each control node ---*/
  for(auto jNode = 0ul; jNode < nCtrlNodesGlobal; jNode++){

    /*--- Distance between non-selected boundary node and control node ---*/
    auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode *nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
    // auto dist = GetDistance(geometry, config, nDim, CtrlCoords[jNode *nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
    // su2double dist;
    // if(config->GetnMarker_Periodic() > 0){
    //     dist = GetDistance(geometry, config, nDim, CtrlCoords[jNode *nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
    //   }else{
    //     dist =  GeometryToolbox::Distance(nDim, CtrlCoords[jNode *nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
    //   }
    /*--- Evaluation of Radial Basis Function ---*/
    auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));

    /*--- Add contribution to error ---*/
    for(auto iDim = 0u; iDim < nDim; iDim++){
      localError[iDim] += rbf*InterpCoeff[jNode*nDim + iDim];
    }
  }


}

void CRadialBasisFunctionInterpolation::GetDoubleEdgeNode(const su2double* maxError, vector<unsigned long>& maxErrorNodes, vector<unsigned short>& maxErrorVector){
  
  /*--- Obtaining maximum error vector and its corresponding angle ---*/
  const auto polarAngleMaxError = atan2(maxError[1],maxError[0]);
  const auto azimuthAngleMaxError = atan2( sqrt( pow(maxError[0],2) + pow(maxError[1],2)), maxError[2]);

  su2double max = 0;
  unsigned long idx;
  unsigned short vec_idx;
  bool found = false;

  for (auto iNodes = 0u; iNodes < BdryNodes.size(); iNodes++){
    for (auto iNode = 0ul; iNode < (*BdryNodes[iNodes]).size(); iNode++){

      auto error = (*BdryNodes[iNodes])[iNode]->GetError();

      su2double polarAngle = atan2(error[1],error[0]);
      su2double relativePolarAngle = abs(polarAngle - polarAngleMaxError);

      switch(nDim){
      case 2:
        if ( abs(relativePolarAngle - M_PI) < M_PI/2 ){
          CompareError(error, iNode, iNodes, max, idx, vec_idx);
        }
        break;
      case 3:

        su2double azimuthAngle = atan2( sqrt( pow(error[0],2) + pow(error[1],2)), error[2]);
        if ( abs(relativePolarAngle - M_PI) < M_PI/2 ){
          if( azimuthAngleMaxError <= M_PI/2 ) {
            if ( azimuthAngle > M_PI/2 - azimuthAngleMaxError) {
              CompareError(error, iNode, iNodes, max, idx, vec_idx);
            }
          }
          else{
            if ( azimuthAngle < 1.5 * M_PI - azimuthAngleMaxError ){
              CompareError(error, iNode, iNodes, max, idx, vec_idx);
            }
          }          
        }
        else{
          if(azimuthAngleMaxError <= M_PI/2){
            if(azimuthAngle > M_PI/2 + azimuthAngleMaxError){
              CompareError(error, iNode, iNodes, max, idx, vec_idx);
            }
          }else{
            if(azimuthAngle < azimuthAngleMaxError -  M_PI/2){
              CompareError(error, iNode, iNodes, max, idx, vec_idx);
            }
          }
        }
        break;
    }
    }
  }

  /*--- Include the found double edge node in the maximum error nodes vector ---*/
  if(max > 0){
    maxErrorNodes.push_back(idx);
    maxErrorVector.push_back(vec_idx);
  }
}

void CRadialBasisFunctionInterpolation::CompareError(su2double* error, unsigned long iNode, unsigned short iNodes, su2double& maxError, unsigned long& idx, unsigned short& vec_idx){
  auto errMag = GeometryToolbox::SquaredNorm(nDim, error);

  if(errMag > maxError){
    maxError = errMag;
    idx = iNode;
    vec_idx = iNodes;
  }
}

void CRadialBasisFunctionInterpolation::SetCorrection(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const vector<unsigned long>& internalNodes, const bool inflationLayer){

  /*--- The non-selected control nodes still have a nonzero error once the maximum error falls below the data reduction tolerance. 
          This error is applied as correction and interpolated into the volumetric mesh for internal nodes that fall within the correction radius.
          To evaluate whether an internal node falls within the correction radius an AD tree is constructed of the boundary nodes,
          making it possible to determine the distance to the nearest boundary node. ---*/

  /*--- Construction of the AD tree consisting of the non-selected boundary nodes ---*/



  /*--- Number of non-selected boundary nodes ---*/
  
  vector<unsigned long> nVertex_prefix_sum(BdryNodes.size()+1, 0);
  for(auto iNodes = 0u; iNodes < BdryNodes.size(); iNodes++){ //TODO change name of iNodes

      nVertex_prefix_sum[iNodes + 1] = nVertex_prefix_sum[iNodes] + BdryNodes[iNodes]->size();
  }
     
  
  /*--- Vector storing the coordinates of the boundary nodes ---*/
  vector<su2double> Coord_bound(nDim*nVertex_prefix_sum.back());

  /*--- Vector storing the IDs of the boundary nodes ---*/
  vector<unsigned long> PointIDs(nVertex_prefix_sum.back());

  /*--- Correction Radius, equal to maximum error times a prescribed constant ---*/
  const su2double CorrectionRadius = config->GetRBF_DataRedCorrectionFactor()*MaxErrorGlobal; 

  /*--- Storing boundary node information ---*/
  unsigned long i = 0;
  for(auto iNodes : BdryNodes){
    for( auto iRBFNode : *iNodes){
      PointIDs[i] = i;
      auto iNode = iRBFNode->GetIndex();
      for(auto iDim = 0u; iDim < nDim; iDim++){
        Coord_bound[i*nDim + iDim] = geometry->nodes->GetCoord(iNode, iDim);
      }
      i++;
    }
  }
  // for(auto iVertex = 0ul; iVertex < nVertexBound; iVertex++){
  //   auto iNode = BoundNodes[iVertex]->GetIndex();
  //   PointIDs[i++] = iVertex;
  //   for(auto iDim = 0u; iDim < nDim; iDim++){
  //     Coord_bound[j++] = geometry->nodes->GetCoord(iNode, iDim);
  //   }
  // }

  /*--- Construction of AD tree ---*/
  CADTPointsOnlyClass BoundADT(nDim, nVertex_prefix_sum.back(), Coord_bound.data(), PointIDs.data(), false);

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
    
    auto idx = distance(nVertex_prefix_sum.begin()+ 1, upper_bound(nVertex_prefix_sum.begin(), nVertex_prefix_sum.end(), pointID)); //TODO change var name
    // if(idx < 0){idx = 0;}
    /*--- Get error of nearest node ---*/
    // auto err = BoundNodes[pointID]->GetError();
    auto err = (*BdryNodes[idx])[pointID-nVertex_prefix_sum[idx]]->GetError();

    /*--- evaluate RBF ---*/    
    auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, CorrectionRadius, dist));

    /*--- Apply correction to the internal node ---*/
    for(auto iDim = 0u; iDim < nDim; iDim++){
      geometry->nodes->AddCoord(internalNodes[iNode], iDim, -rbf*err[iDim]);
    }
  }
  


  /*--- Applying the correction to the non-selected boundary nodes ---*/
  // for(auto iNode = 0ul; iNode < BoundNodes.size(); iNode++){
  //   auto err =  BoundNodes[iNode]->GetError();
  //   for(auto iDim = 0u; iDim < nDim; iDim++){
  //     geometry->nodes->AddCoord(BoundNodes[iNode]->GetIndex(), iDim, -err[iDim]);
  //   }
  // }
  if(inflationLayer){
    BdryNodes.resize(1); // todo if statement in case its for the inflation layer
  }
  for(auto iNodes : BdryNodes){
    for (auto iNode : *iNodes){
      auto err = iNode->GetError();
      for(auto iDim=0; iDim < nDim; iDim++){
        geometry->nodes->AddCoord(iNode->GetIndex(), iDim, -err[iDim]);
      }
    }
  }
}


void CRadialBasisFunctionInterpolation::AddControlNode(vector<unsigned short>& maxErrorVector, vector<unsigned long>& maxErrorNodes, su2double& maxErrorLocal){
  

  // /*--- Sort indices in descending order, to prevent shift in index as they are erased ---*/
  if(maxErrorNodes.size() == 2 && maxErrorVector[0] == maxErrorVector[1]){
    sort(maxErrorNodes.rbegin(),maxErrorNodes.rend()); 
  }
  
  for(auto iNode = 0u; iNode < maxErrorNodes.size(); iNode++){
    ControlNodes[maxErrorVector[iNode]]->push_back((*BdryNodes[maxErrorVector[iNode]])[maxErrorNodes[iNode]]);
    // /*--- Addition of node to the reduced set of control nodes ---*/
    // ReducedControlNodes.push_back(move(BoundNodes[iNode]));

    /*--- Removal of node among the non-selected boundary nodes ---*/
    // BoundNodes.erase(BoundNodes.begin()+iNode);
    BdryNodes[maxErrorVector[iNode]]->erase(BdryNodes[maxErrorVector[iNode]]->begin() + maxErrorNodes[iNode]);
  }

  Get_nCtrlNodes();

  /*--- Clearing maxErrorNodes vector ---*/
  maxErrorNodes.clear();
  maxErrorVector.clear();

 
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


void CRadialBasisFunctionInterpolation::GetIL_EdgeDeformation(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, unsigned short iLayer){
  
  // the assumption is made that all control nodes of the inflation layer are part of the same marker.
  // if this is no longer the case then this requires a switch back to the combination of selected and
  // non selected control nodes


  /*--- Number of deformation steps ---*/ //TODO these are also defined in the GetIL_EdgeVar function
  auto Nonlinear_iter = config->GetGridDef_Nonlinear_Iter();
  auto wallMarkerIdx = (*ControlNodes[0])[0]->GetMarker();
  auto nVertex = geometry->GetnVertex(wallMarkerIdx);


  /*--- Obtaining free displacement of edge nodes ---*/
  GetFreeDeformation(geometry, config, type, radius, test_edge[iLayer]);
  if (red_edge[iLayer]->size() != 0){
    GetFreeDeformation(geometry, config, type, radius, red_edge[iLayer]);
  }

  /*--- Loop over inflation layer wall nodes ---*/  
  GetIL_EdgeVar(geometry, config, iLayer);

 
  
  /*--- Set wall back to initial position for accurate calculation of RBFs ---*/
for( auto jNode = 0ul; jNode < nVertex ; jNode++){

    auto var_coord = geometry->vertex[wallMarkerIdx][jNode]->GetVarCoord();
    for(auto iDim = 0u; iDim < nDim; iDim++){

      /*--- Applying the deformation ---*/
      geometry->nodes->AddCoord(geometry->vertex[wallMarkerIdx][jNode]->GetNode(), iDim, -var_coord[iDim]/Nonlinear_iter);

    }
  }

  /*--- Update wall boundary ---*/
  geometry->SetBoundControlVolume(config, UPDATE);
}

void CRadialBasisFunctionInterpolation::SetNodes(vector<CRadialBasisFunctionNode*>* reducedNodes, vector<CRadialBasisFunctionNode*>* Nodes, unsigned short index){
  
  if(ControlNodes.size() != index+1){
    ControlNodes.resize(index+1);
  }
  if(BdryNodes.size() != index+1){
    BdryNodes.resize(index+1);
  }
  ControlNodes[index] = reducedNodes;
  BdryNodes[index] = Nodes;
  
  Get_nCtrlNodes();

}

void CRadialBasisFunctionInterpolation::ResetError(vector<unsigned long>& maxErrorNodes, vector<unsigned short>& maxErrorVector, su2double& maxErrorLocal){
  maxErrorLocal = 0;
  maxErrorNodes.clear();
  maxErrorVector.clear();
}

//TODO rename function
void CRadialBasisFunctionInterpolation::GetFreeDeformation(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, vector<CRadialBasisFunctionNode*>* targetNodes){

    /*--- Obtaining free displacement of the inflation layer edge nodes ---*/
  for ( auto iNode : *targetNodes){
    
    /*--- Obtaining coordinates ---*/
    auto coord = geometry->nodes->GetCoord(iNode->GetIndex());

    /*--- Setting new coord equal to old coord ---*/
    iNode->SetNewCoord(coord, nDim);

    /*--- Loop for contribution of each control node ---*/
    for(auto jNode = 0ul; jNode < nCtrlNodesGlobal; jNode++){
    
      /*--- Determine distance between considered internal and control node ---*/
      auto dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
      // auto dist = GetDistance(geometry, config, nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
      // su2double dist;
      // if(config->GetnMarker_Periodic() > 0){
      //   dist = GetDistance(geometry, config, nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
      // }else{
      //   dist = GeometryToolbox::Distance(nDim, CtrlCoords[jNode*nDim], geometry->nodes->GetCoord(iNode->GetIndex()));
      // }
      /*--- Evaluate RBF based on distance ---*/
      auto rbf = SU2_TYPE::GetValue(CRadialBasisFunction::Get_RadialBasisValue(type, radius, dist));
      
      /*--- Add contribution to new coordinates -- -*/
      for(auto iDim = 0u; iDim < nDim; iDim++){
        iNode->AddNewCoord(rbf*InterpCoeff[jNode*nDim+iDim], iDim);
      }
    }
  }  
}

void CRadialBasisFunctionInterpolation::GetIL_EdgeVar(CGeometry* geometry, CConfig* config, unsigned short iLayer){
  auto wallMarkerIdx = (*ControlNodes[0])[0]->GetMarker();
  const auto Nonlinear_iter = config->GetGridDef_Nonlinear_Iter();
  auto nVertex = geometry->GetnVertex(wallMarkerIdx);

  /*--- Assembly of AD tree containing the updated wall nodes ---*/
  vector<su2double> Coord_bound(nDim * nVertex);
  vector<unsigned long> PointIDs(nVertex);
  unsigned long ii = 0;

  /*--- Loop through wall nodes ---*/
  for( auto jNode = 0ul; jNode < nVertex ; jNode++){

    /*--- Assign identifier ---*/
    PointIDs[jNode] = jNode;

    auto var_coord = geometry->vertex[wallMarkerIdx][jNode]->GetVarCoord();
    for(auto iDim = 0u; iDim < nDim; iDim++){

      /*--- Applying the deformation ---*/
      // geometry->nodes->AddCoord((*ControlNodes[0])[jNode]->GetIndex(), iDim, CtrlNodeDeformation[jNode * nDim + iDim]);
      geometry->nodes->AddCoord(geometry->vertex[wallMarkerIdx][jNode]->GetNode(), iDim, var_coord[iDim]/Nonlinear_iter);
      /*--- store updated position ---*/
      // Coord_bound[ii++] = geometry->nodes->GetCoord((*ControlNodes[0])[jNode]->GetIndex())[iDim];
      Coord_bound[ii++] = geometry->nodes->GetCoord(geometry->vertex[wallMarkerIdx][jNode]->GetNode())[iDim];
    }
  }

  /*--- Update of boundary to obtain normals of updated geometry ---*/
  geometry->SetBoundControlVolume(config, UPDATE);

  /*--- AD tree with updated wall positions ---*/
  CADTPointsOnlyClass WallADT(nDim, nVertex, Coord_bound.data(), PointIDs.data(), true);


   /*--- Inflation layer height ---*/
  su2double IL_height = config->GetRBF_IL_Height();
  

  unsigned long pointID;
  int rankID;
  su2double dist; 
  
  /*--- Finding the required displacement of the edge nodes ---*/

  /*--- Distance to nearest wall node and required added inflation layer thickness ---*/
  su2double dist_vec[nDim];
  su2double added_thickness;

  /*--- Loop over inflation layer wall nodes ---*/
       
  for ( auto iNode : *test_edge[iLayer]){
    /*--- Get nearest wall node ---*/
    WallADT.DetermineNearestNode(iNode->GetNewCoord(), dist, pointID, rankID);

    /*--- Get normal and make it a unit vector ---*/
    auto normal = geometry->vertex[wallMarkerIdx][pointID]->GetNormal(); 
    auto normal_length = GeometryToolbox::Norm(nDim, normal);
    for(auto iDim = 0u; iDim < nDim; iDim++){
      normal[iDim] = normal[iDim]/normal_length;
    }

    /*--- Get distance vector from edge node to nearest wall node ---*/
    GeometryToolbox::Distance(nDim, iNode->GetNewCoord(), geometry->nodes->GetCoord(geometry->vertex[wallMarkerIdx][pointID]->GetNode()), dist_vec);

    /*--- Dot product to obtain current inflation layer height ---*/
    auto dp = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

    /*--- Get required change in inflation layer thickness 
      sign changes depending on whether the normal is pointing outward/inward ---*/
    if (dp >= 0){
      added_thickness = + IL_height - abs(dp); 
    }else{
      added_thickness = - IL_height + abs(dp);
    }

    /*--- Apply required change in coordinates and store variation w.r.t. initial coordinates. ---*/
    su2double var_coord[nDim];
    for(auto iDim = 0u; iDim < nDim; iDim++){
      iNode->AddNewCoord(added_thickness * normal[iDim], iDim);
      var_coord[iDim] = (iNode->GetNewCoord()[iDim] - geometry->nodes->GetCoord(iNode->GetIndex())[iDim])*Nonlinear_iter;
    }

    iNode->SetVarCoord(var_coord, nDim);
  }

  for ( auto iNode : *red_edge[iLayer]){
    /*--- Get nearest wall node ---*/
    WallADT.DetermineNearestNode(iNode->GetNewCoord(), dist, pointID, rankID);

    /*--- Get normal and make it a unit vector ---*/
    auto normal = geometry->vertex[wallMarkerIdx][pointID]->GetNormal(); 
    auto normal_length = GeometryToolbox::Norm(nDim, normal);
    for(auto iDim = 0u; iDim < nDim; iDim++){
      normal[iDim] = normal[iDim]/normal_length;
    }

    /*--- Get distance vector from edge node to nearest wall node ---*/
    GeometryToolbox::Distance(nDim, iNode->GetNewCoord(), geometry->nodes->GetCoord(geometry->vertex[wallMarkerIdx][pointID]->GetNode()), dist_vec);
    
    /*--- Dot product to obtain current inflation layer height ---*/
    auto dp = GeometryToolbox::DotProduct(nDim, normal, dist_vec);

    /*--- Get required change in inflation layer thickness 
      sign changes depending on whether the normal is pointing outward/inward ---*/
    if (dp >= 0){
      added_thickness = + IL_height - abs(dp); 
    }else{
      added_thickness = - IL_height + abs(dp);
    }

    /*--- Apply required change in coordinates and store variation w.r.t. initial coordinates. ---*/
    su2double var_coord[nDim];
    for(auto iDim = 0u; iDim < nDim; iDim++){
      iNode->AddNewCoord(added_thickness * normal[iDim], iDim);
      var_coord[iDim] = (iNode->GetNewCoord()[iDim] - geometry->nodes->GetCoord(iNode->GetIndex())[iDim])*Nonlinear_iter;
    }

    iNode->SetVarCoord(var_coord, nDim);
  }

}

su2double CRadialBasisFunctionInterpolation::GetDistance(CGeometry* geometry, CConfig* config, unsigned short nDim, const su2double *a, const su2double *b){
  // if (config->GetnMarker_Periodic() > 0){
    // const su2double* per_translation = config->GetPeriodic_Translation(0);
    // su2double dist(0.0);

    // for (auto iDim = 0u; iDim < nDim; iDim++){
    //   if (per_translation[iDim] != 0){
    //     dist += pow( (per_translation[iDim]/M_PI*sin((a[iDim] - b[iDim]) * M_PI/per_translation[iDim] )),2);
    //   } else{ dist += pow( a[iDim] - b[iDim], 2); }
    // }
    
    // return sqrt(dist); 
  // }else{
  //   return GeometryToolbox::Distance(nDim, a, b);
  // }
  return -1;
}