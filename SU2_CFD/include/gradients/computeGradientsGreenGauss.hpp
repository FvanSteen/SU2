/*!
 * \file computeGradientsGreenGauss.hpp
 * \brief Generic implementation of Green-Gauss gradient computation.
 * \note This allows the same implementation to be used for conservative
 *       and primitive variables of any solver.
 * \author P. Gomes
 * \version 8.0.0 "Harrier"
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

#include <vector>
#include <algorithm>

#include "../../../Common/include/parallelization/omp_structure.hpp"
#include "../../../Common/include/toolboxes/geometry_toolbox.hpp"

namespace detail {

// find local vertex on a symmetry marker using global iPoint
inline su2double* getVertexNormalfromPoint(const CConfig& config, CGeometry& geometry, unsigned long iPointGlobal){
  unsigned long iPointSym=0;
  for (size_t iMarker = 0; iMarker < geometry.GetnMarker(); ++iMarker) {
    if (config.GetMarker_All_KindBC(iMarker) == SYMMETRY_PLANE) {
      for (size_t iVertex = 0; iVertex < geometry.GetnVertex(iMarker); ++iVertex) {
        iPointSym = geometry.vertex[iMarker][iVertex]->GetNode();
        if (iPointSym == iPointGlobal)
          return geometry.vertex[iMarker][iVertex]->GetNormal();
      }
    }
  }
  cout << "point is not found " << endl;
  exit(0);
}

/*!
 * \brief Compute the gradient of a field using the Green-Gauss theorem.
 * \ingroup FvmAlgos
 * \note Template nDim to allow efficient unrolling of inner loops.
 * \note Gradients can be computed only for a contiguous range of variables, defined
 *       by [varBegin, varEnd[ (e.g. 0,1 computes the gradient of the 1st variable).
 *       This can be used, for example, to compute only velocity gradients.
 * \note The function uses an optional solver object to perform communications, if
 *       none (nullptr) is provided the function does not fail (the objective of
 *       this is to improve test-ability).
 * \param[in] solver - Optional, solver associated with the field (used only for MPI).
 * \param[in] kindMpiComm - Type of MPI communication required.
 * \param[in] kindPeriodicComm - Type of periodic communication required.
 * \param[in] geometry - Geometric grid properties.
 * \param[in] config - Configuration of the problem, used to identify types of boundaries.
 * \param[in] field - Generic object implementing operator (iPoint, iVar).
 * \param[in] varBegin - Index of first variable for which to compute the gradient.
 * \param[in] varEnd - Index of last variable for which to compute the gradient.
 * \param[out] gradient - Generic object implementing operator (iPoint, iVar, iDim).
 */
template <size_t nDim, class FieldType, class GradientType>
void computeGradientsGreenGauss(CSolver* solver, MPI_QUANTITIES kindMpiComm, PERIODIC_QUANTITIES kindPeriodicComm,
                                CGeometry& geometry, const CConfig& config, const FieldType& field, size_t varBegin,
                                size_t varEnd, GradientType& gradient) {
  const size_t nPointDomain = geometry.GetnPointDomain();

#ifdef HAVE_OMP
  constexpr size_t OMP_MAX_CHUNK = 512;

  const auto chunkSize = computeStaticChunkSize(nPointDomain, omp_get_max_threads(), OMP_MAX_CHUNK);
#endif

  static constexpr size_t MAXNVAR = 20;

  /*--- For each (non-halo) volume integrate over its faces (edges). ---*/

  SU2_OMP_FOR_DYN(chunkSize)
  for (size_t iPoint = 0; iPoint < nPointDomain; ++iPoint) {
    auto nodes = geometry.nodes;

    /*--- Cannot preaccumulate if hybrid parallel due to shared reading. ---*/
    if (omp_get_num_threads() == 1) AD::StartPreacc();
    AD::SetPreaccIn(nodes->GetVolume(iPoint));
    AD::SetPreaccIn(nodes->GetPeriodicVolume(iPoint));

    for (size_t iVar = varBegin; iVar < varEnd; ++iVar) AD::SetPreaccIn(field(iPoint, iVar));

    /*--- Clear the gradient. --*/

    for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
      for (size_t iDim = 0; iDim < nDim; ++iDim) gradient(iPoint, iVar, iDim) = 0.0;

    /*--- Handle averaging and division by volume in one constant. ---*/

    su2double halfOnVol = 0.5 / (nodes->GetVolume(iPoint) + nodes->GetPeriodicVolume(iPoint));

    /*--- Add a contribution due to each neighbor. ---*/

    for (size_t iNeigh = 0; iNeigh < nodes->GetnPoint(iPoint); ++iNeigh) {
      size_t iEdge = nodes->GetEdge(iPoint, iNeigh);
      size_t jPoint = nodes->GetPoint(iPoint, iNeigh);

      /*--- Determine if edge points inwards or outwards of iPoint.
       *    If inwards we need to flip the area vector. ---*/

      su2double dir = (iPoint < jPoint) ? 1.0 : -1.0;
      su2double weight = dir * halfOnVol;

      const auto area = geometry.edges->GetNormal(iEdge);
      AD::SetPreaccIn(area, nDim);

      for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
        AD::SetPreaccIn(field(jPoint, iVar));
        su2double flux = weight * (field(iPoint, iVar) + field(jPoint, iVar));

        for (size_t iDim = 0; iDim < nDim; ++iDim) gradient(iPoint, iVar, iDim) += flux * area[iDim];
      }
    }

    for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
      for (size_t iDim = 0; iDim < nDim; ++iDim) AD::SetPreaccOut(gradient(iPoint, iVar, iDim));

    AD::EndPreacc();
  }
  END_SU2_OMP_FOR

  /* For symmetry planes, we need to impose the conditions (Blazek eq. 8.40):
   * 1. n.grad(phi) = 0
   * 2. n.grad(v.t) = 0
   * 3. t.grad(v.n) = 0
   */
  su2double flux[MAXNVAR] = {0.0};
  su2double fluxReflected[MAXNVAR] = {0.0};

  for (size_t iMarker = 0; iMarker < geometry.GetnMarker(); ++iMarker) {
    if (config.GetMarker_All_KindBC(iMarker) == SYMMETRY_PLANE) {
      for (size_t iVertex = 0; iVertex < geometry.GetnVertex(iMarker); ++iVertex) {

        size_t iPoint = geometry.vertex[iMarker][iVertex]->GetNode();
        auto nodes = geometry.nodes;
        // we need to set the gradient to zero for the entire marker to prevent double-counting
        // points that are shared by other markers
        for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
          for (size_t iDim = 0; iDim < nDim; ++iDim) gradient(iPoint, iVar, iDim) = 0.0;

        su2double halfOnVol = 0.5 / (nodes->GetVolume(iPoint) + nodes->GetPeriodicVolume(iPoint));
        for (size_t iNeigh = 0; iNeigh < nodes->GetnPoint(iPoint); ++iNeigh) {
          size_t iEdge = nodes->GetEdge(iPoint, iNeigh);
          size_t jPoint = nodes->GetPoint(iPoint, iNeigh);
          su2double* icoord = nodes->GetCoord(iPoint); // nijso: debugging
          su2double* jcoord = nodes->GetCoord(jPoint); // nijso: debugging

          //su2double Veli[nDim] = {0.0};
          //for (size_t iDim = 0; iDim < nDim; ++iDim) Veli[iDim] = field(iPoint, iDim + 1);
          //su2double Velj[nDim] = {0.0};
          //for (size_t iDim = 0; iDim < nDim; ++iDim) Velj[iDim] = field(jPoint, iDim + 1);

          /*--- Determine if edge points inwards or outwards of iPoint.
           *    If inwards we need to flip the area vector. ---*/

          su2double dir = (iPoint < jPoint) ? 1.0 : -1.0;
          su2double weight = dir * halfOnVol;
          const auto area = geometry.edges->GetNormal(iEdge);

         /*--- Normal vector for this vertex (negate for outward convention). ---*/
          const su2double* VertexNormal = geometry.vertex[iMarker][iVertex]->GetNormal();

          // reflected normal V=U - 2U_t
          const auto NormArea = GeometryToolbox::Norm(nDim, VertexNormal);
          su2double UnitNormal[nDim] = {0.0};
          for (size_t iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = VertexNormal[iDim] / NormArea;
          su2double ProjArea = 0.0;
          for (unsigned long iDim = 0; iDim < nDim; iDim++) ProjArea += area[iDim] * UnitNormal[iDim];
          su2double areaReflected[nDim] = {0.0};
          for (size_t iDim = 0; iDim < nDim; iDim++)
            areaReflected[iDim] = area[iDim] - 2.0 * ProjArea * UnitNormal[iDim];

          for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
            flux[iVar] = weight * (field(iPoint, iVar) + field(jPoint, iVar));
            fluxReflected[iVar] = flux[iVar];
          }

          su2double ProjFlux = 0.0;
          for (size_t iDim = 0; iDim < nDim; iDim++) ProjFlux += flux[iDim + 1] * UnitNormal[iDim];

          for (size_t iDim = 0; iDim < nDim; iDim++)
            fluxReflected[iDim + 1] = flux[iDim + 1] - 2.0 * ProjFlux * UnitNormal[iDim];

          for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
            for (size_t iDim = 0; iDim < nDim; ++iDim) {
              gradient(iPoint, iVar, iDim) += 0.5 * (flux[iVar] * area[iDim] + fluxReflected[iVar] *
               areaReflected[iDim]);

            }
          }

        } // loop over the edges

         // /*--- For nodes shared with walls, we can simply add the mirrored contribution. The nonmirrored
          //  *   contributions is added
          //  *    in the routine below. ---*/
          // if (nodes->GetSolidBoundary(iPoint)) {
          //   cout << "point shared by symmetry and wall" << endl;
          //   su2double volume = nodes->GetVolume(iPoint) + nodes->GetPeriodicVolume(iPoint);
          //   const auto area = geometry.vertex[iMarker][iVertex]->GetNormal();
          //   const su2double* VertexNormal = geometry.vertex[iMarker][iVertex]->GetNormal();
          //   const auto NormArea = GeometryToolbox::Norm(nDim, VertexNormal);
          //   su2double UnitNormal[nDim] = {0.0};
          //   for (size_t iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = VertexNormal[iDim] / NormArea;
          //   su2double ProjArea = 0.0;
          //   for (unsigned long iDim = 0; iDim < nDim; iDim++) ProjArea += area[iDim] * UnitNormal[iDim];
          //   su2double areaReflected[nDim] = {0.0};
          //   for (size_t iDim = 0; iDim < nDim; iDim++)
          //     areaReflected[iDim] = area[iDim] - 2.0 * ProjArea * UnitNormal[iDim];

          //   /*--- First, use the values at node i only (better to use entire face but we do not have it) ---*/
          //   for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
          //     flux[iVar] = field(iPoint, iVar) / volume;
          //     fluxReflected[iVar] = flux[iVar];
          //   }
          //   /*--- project the flux ---*/
          //   su2double ProjFlux = 0.0;
          //   for (size_t iDim = 0; iDim < nDim; iDim++) ProjFlux += flux[iDim + 1] * UnitNormal[iDim];

          //   for (size_t iDim = 0; iDim < nDim; iDim++)
          //     fluxReflected[iDim + 1] = flux[iDim + 1] - 2.0 * ProjFlux * UnitNormal[iDim];

          //   for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
          //     for (size_t iDim = 0; iDim < nDim; ++iDim) {
          //       gradient(iPoint, iVar, iDim) -= 0.5*fluxReflected[iVar] * areaReflected[iDim];
          //       gradient(iPoint, iVar, iDim) += 0.5*flux[iVar] * area[iDim];
          //     }
          //   }
          // }

          // // // /*--- check if point is shared with an inlet, outlet or far_field ---*/
          // if (nodes->Getinoutfar(iPoint)) {
          //   cout << "point = " << iPoint << endl;
          //   su2double volume = nodes->GetVolume(iPoint) + nodes->GetPeriodicVolume(iPoint);
          //   const auto area = geometry.vertex[iMarker][iVertex]->GetNormal();
          //   const su2double* VertexNormal = geometry.vertex[iMarker][iVertex]->GetNormal();
          //   const auto NormArea = GeometryToolbox::Norm(nDim, VertexNormal);
          //   su2double UnitNormal[nDim] = {0.0};
          //   for (size_t iDim = 0; iDim < nDim; iDim++) UnitNormal[iDim] = VertexNormal[iDim] / NormArea;
          //   su2double ProjArea = 0.0;
          //   for (unsigned long iDim = 0; iDim < nDim; iDim++) ProjArea += area[iDim] * UnitNormal[iDim];
          //   su2double areaReflected[nDim] = {0.0};
          //   for (size_t iDim = 0; iDim < nDim; iDim++)
          //     areaReflected[iDim] = area[iDim] - 2.0 * ProjArea * UnitNormal[iDim];

          //   /*--- First, use the values at node i only (better to use entire face but we do not have it) ---*/
          //   for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
          //     // note that we have 2x the regular volume
          //     // we use 1x the regular volume in the other routine so we have
          //     // (1/2)*(1/Volume) * (F*a + Fr*ar) = (1/2)*(1/Volume)*(Fr*ar) + 1/volume*(F*a) - (1/2)*
          //     flux[iVar] = field(iPoint, iVar) / volume;
          //     fluxReflected[iVar] = flux[iVar];
          //   }
          //   /*--- project the flux ---*/
          //   su2double ProjFlux = 0.0;
          //   for (size_t iDim = 0; iDim < nDim; iDim++) ProjFlux += flux[iDim + 1] * UnitNormal[iDim];

          //   for (size_t iDim = 0; iDim < nDim; iDim++)
          //     fluxReflected[iDim + 1] = flux[iDim + 1] - 2.0 * ProjFlux * UnitNormal[iDim];

          //   for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
          //     for (size_t iDim = 0; iDim < nDim; ++iDim) {
          //       cout << "before: " << iPoint << ", " << iVar << ", " << iDim << ", " << gradient(iPoint,iVar,iDim)
          //            << ", delta_L: " <<  flux[iVar] <<", "<< area[iDim]
          //            << ", delta_R: " <<  fluxReflected[iVar] <<", "<< areaReflected[iDim]
          //            << ", " << gradient(iPoint,iVar,iDim) - fluxReflected[iVar] * areaReflected[iDim] << endl;
          //       gradient(iPoint, iVar, iDim) -= 0.5*fluxReflected[iVar] * areaReflected[iDim];
          //       // below we subtract 1/V, but we need only 0.5/V, so we add 0.5*V here
          //       gradient(iPoint, iVar, iDim) += 0.5*flux[iVar] * area[iDim];
          //     }
          //   }
          // }


      } //ivertex
    } //symmetry
  } //loop over markers


  for (size_t iMarker = 0; iMarker < geometry.GetnMarker(); ++iMarker) {
    if ((config.GetMarker_All_KindBC(iMarker) != INTERNAL_BOUNDARY) &&
        (config.GetMarker_All_KindBC(iMarker) != NEARFIELD_BOUNDARY) &&
        (config.GetMarker_All_KindBC(iMarker) != SYMMETRY_PLANE) &&
        (config.GetMarker_All_KindBC(iMarker) != PERIODIC_BOUNDARY)) {
      /*--- Work is shared in inner loop as two markers
       *    may try to update the same point. ---*/

      SU2_OMP_FOR_STAT(32)
      for (size_t iVertex = 0; iVertex < geometry.GetnVertex(iMarker); ++iVertex) {
        size_t iPoint = geometry.vertex[iMarker][iVertex]->GetNode();
        auto nodes = geometry.nodes;

        /*--- Halo points do not need to be considered. ---*/

        if (!nodes->GetDomain(iPoint)) continue;

        su2double volume = nodes->GetVolume(iPoint) + nodes->GetPeriodicVolume(iPoint);
        const auto area = geometry.vertex[iMarker][iVertex]->GetNormal();
        for (size_t iVar = varBegin; iVar < varEnd; iVar++)
          flux[iVar] = field(iPoint,iVar) / volume;

        // When the node is shared with a symmetry we need to mirror the contribution of
        // the face that is coincident with the inlet/outlet
        if (nodes->GetSymmetry(iPoint) && nodes->Getinoutfar(iPoint)) {
          cout << "iPoint "
               << iPoint
               << " is on a symmetry plane and an inlet/outlet"
               << nodes->GetSymmetry(iPoint)
               << ", "
               << nodes->Getinoutfar(iPoint)<< endl;

          // we have to find the edges that were missing in the symmetry computations.
          // So we find the jPoints that are on the inlet plane
          // so we loop over all neighbor of iPoint, find all jPoints and then check if it is on the inlet
          for (size_t iNeigh = 0; iNeigh < nodes->GetnPoint(iPoint); ++iNeigh) {
            size_t iEdge = nodes->GetEdge(iPoint, iNeigh);
            size_t jPoint = nodes->GetPoint(iPoint, iNeigh);
            if (nodes->Getinoutfar(jPoint)) {
              cout << "  jPoint " << jPoint << " is on the inlet plane" << endl;
              // this edge jPoint - jPoint is the missing edge for the symmetry computations
              //compute the flux on the face between iPoint and jPoint
              //for (size_t iVar = varBegin; iVar < varEnd; iVar++) {
              //  flux[iVar] = 0.5*(field(iPoint,iVar) + field(jPoint, iVar)) / (2.0*volume);
            }
            if (nodes->GetSymmetry(jPoint)) {
              cout << "  jPoint " << jPoint << " is on the symmetry plane" << endl;
              // this edge iPoint - jPoint is the missing edge for the symmetry computations
              // we now need to get the normal of the symmetry plane at jpoint.
              // so we loop over the markers, find all symmetry planes, check if the ipoint is on the plane
              const su2double* VertexNormal = getVertexNormalfromPoint(config, geometry,jPoint);
              cout << "  vertex normal = " << VertexNormal[0] <<", " << VertexNormal[1] << endl;
              // get the normal on the vertex

              // now reflect in the mirror
              // reflected normal V=U - 2U_t
              const auto NormArea = GeometryToolbox::Norm(nDim, VertexNormal);
              su2double UnitNormal[nDim] = {0.0};
              for (size_t iDim = 0; iDim < nDim; iDim++)
                UnitNormal[iDim] = VertexNormal[iDim] / NormArea;
              su2double ProjArea = 0.0;
              for (unsigned long iDim = 0; iDim < nDim; iDim++)
                ProjArea += area[iDim] * UnitNormal[iDim];
              su2double areaReflected[nDim] = {0.0};
              for (size_t iDim = 0; iDim < nDim; iDim++)
                areaReflected[iDim] = area[iDim] - 2.0 * ProjArea * UnitNormal[iDim];

              for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
                flux[iVar] = 0.5 * (field(iPoint, iVar) + field(jPoint, iVar)) / (2.0*volume);
                fluxReflected[iVar] = flux[iVar];
              }

              su2double ProjFlux = 0.0;
              for (size_t iDim = 0; iDim < nDim; iDim++)
                ProjFlux += flux[iDim + 1] * UnitNormal[iDim];

              for (size_t iDim = 0; iDim < nDim; iDim++)
                fluxReflected[iDim + 1] = flux[iDim + 1] - 2.0 * ProjFlux * UnitNormal[iDim];

              for (size_t iVar = varBegin; iVar < varEnd; ++iVar) {
                for (size_t iDim = 0; iDim < nDim; ++iDim) {
                  gradient(iPoint, iVar, iDim) += 0.5 * (flux[iVar] * area[iDim] + fluxReflected[iVar] *
                   areaReflected[iDim]);
                }
              }

            } // if symmetry
          } //neighbors

        } else {
          // if we are on a marker but not on a share point between a symmetry and an inlet/outlet
          for (size_t iVar = varBegin; iVar < varEnd; iVar++) {
            for (size_t iDim = 0; iDim < nDim; iDim++) {
              gradient(iPoint, iVar, iDim) -= flux[iVar] * area[iDim];
            }
          } // loop over variables
        } // symmetry and in/out shared node
      } // vertices
      END_SU2_OMP_FOR
    } //found right marker
  } // iMarkers

  /*--- If no solver was provided we do not communicate ---*/

  if (solver == nullptr) return;

  /*--- Account for periodic contributions. ---*/

  for (size_t iPeriodic = 1; iPeriodic <= config.GetnMarker_Periodic() / 2; ++iPeriodic) {
    solver->InitiatePeriodicComms(&geometry, &config, iPeriodic, kindPeriodicComm);
    solver->CompletePeriodicComms(&geometry, &config, iPeriodic, kindPeriodicComm);
  }

  /*--- Obtain the gradients at halo points from the MPI ranks that own them. ---*/

  solver->InitiateComms(&geometry, &config, kindMpiComm);
  solver->CompleteComms(&geometry, &config, kindMpiComm);
}
}  // namespace detail



/*!
 * \brief Instantiations for 2D and 3D.
 * \ingroup FvmAlgos
 */
template <class FieldType, class GradientType>
void computeGradientsGreenGauss(CSolver* solver, MPI_QUANTITIES kindMpiComm, PERIODIC_QUANTITIES kindPeriodicComm,
                                CGeometry& geometry, const CConfig& config, const FieldType& field, size_t varBegin,
                                size_t varEnd, GradientType& gradient) {
  switch (geometry.GetnDim()) {
    case 2:
      detail::computeGradientsGreenGauss<2>(solver, kindMpiComm, kindPeriodicComm, geometry, config, field, varBegin,
                                            varEnd, gradient);
      break;
    case 3:
      detail::computeGradientsGreenGauss<3>(solver, kindMpiComm, kindPeriodicComm, geometry, config, field, varBegin,
                                            varEnd, gradient);
      break;
    default:
      SU2_MPI::Error("Too many dimensions to compute gradients.", CURRENT_FUNCTION);
      break;
  }
}
