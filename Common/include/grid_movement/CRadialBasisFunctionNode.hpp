/*!
 * \file CRadialBasisFunctionNode.hpp
 * \brief Declaration of the RBF node class that stores nodal information for the RBF interpolation.
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

#pragma once

#include "../../../Common/include/geometry/CGeometry.hpp"

/*!
 * \class CRadialBasisFunctionNode
 * \brief Class for defining a Radial Basis Function Node
 * \author F. van Steen
 */

class CRadialBasisFunctionNode{
 protected:
  unsigned long idx;          /*!< \brief Global index. */
  unsigned short marker_idx;  /*!< \brief Marker index. */
  unsigned long vertex_idx;   /*!< \brief Vertex index. */
    
  su2double error[3];         /*!< \brief Nodal data reduction error. */
  su2double new_coord[3];     /*!< \brief New coordinate position. */
  su2double var_coord[3];     /*!< \brief Variation in coordinates. */
    
  public:

  /*!
  * \brief Constructor of the class.
  * \param[in] idx_val - Local node index.
  * \param[in] marker_val - Local marker index.
  * \param[in] vertex_val - Local vertex index.
  */
  CRadialBasisFunctionNode(unsigned long idx_val, unsigned short marker_val, unsigned long vertex_val);

  /*!
  * \brief Returns local global index.
  * \return Local node index.
  */
  inline unsigned long GetIndex(){return idx;}

  /*!
  * \brief Returns local vertex index.
  * \return Local vertex index.
  */
  inline unsigned long GetVertex(){return vertex_idx;}

  /*!
  * \brief Returns local marker index.
  * \return Local marker index.
  */
  inline unsigned short GetMarker(){return marker_idx;}

  /*!
  * \brief Set error of the RBF node.
  * \param val_error - Nodal error.
  * \param nDim - Number of dimensions.
  */

  inline void SetError(const su2double* val_error, unsigned short nDim) {
    for (auto iDim = 0u; iDim < nDim; iDim++) error[iDim] = val_error[iDim];
  }

  /*!
  * \brief Get nodal error.
  * \return Nodal error.
  */
  inline su2double* GetError(){ return error;}

  //TODO add descriptions

  inline void SetNewCoord(const su2double* coord, unsigned short nDim) {
    for (auto iDim = 0u; iDim < nDim; iDim++) new_coord[iDim] = coord[iDim];
  }

  inline void AddNewCoord(su2double coord, unsigned short iDim) {new_coord[iDim] += coord;}

  inline su2double* GetNewCoord(){ return new_coord;}

  inline void SetVarCoord(const su2double* coord, unsigned short nDim) {
    for (auto iDim = 0u; iDim < nDim; iDim++) var_coord[iDim] = coord[iDim];
  }

  inline su2double* GetVarCoord(){ return var_coord;}


};