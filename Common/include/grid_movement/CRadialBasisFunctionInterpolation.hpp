/*!
 * \file CRadialBasisFunctionInterpolation.hpp
 * \brief Headers of the CRadialBasisFunctionInterpolation class.
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
#include "CVolumetricMovement.hpp"
#include "CRadialBasisFunctionNode.hpp"
#include "../../include/toolboxes/CSymmetricMatrix.hpp"
#include "../../include/adt/CADTPointsOnlyClass.hpp"

/*!
 * \class CRadialBasisFunctionInterpolation
 * \brief Class for moving the volumetric numerical grid using Radial Basis Function interpolation.
 * \author F. van Steen
 */

class CRadialBasisFunctionInterpolation : public CVolumetricMovement {
protected:

  vector<vector<CRadialBasisFunctionNode*>*> ControlNodes; /*!< \brief Vector with control nodes*/
  vector<CRadialBasisFunctionNode*> BoundNodes;               /*!< \brief Vector with boundary nodes.*/
  vector<CRadialBasisFunctionNode*> ReducedBoundNodes;      /*!< \brief Vector with selected control nodes in data reduction algorithm. */
  vector<CRadialBasisFunctionNode*> ReducedSlideNodes;      /*!< \brief Vector with selected control nodes in data reduction algorithm. */
  vector<CRadialBasisFunctionNode*> SlideSurfNodes;
  
  vector<su2double> CtrlNodeDeformation;  /*!< \brief Control Node Deformation.*/ 
  vector<su2double> InterpCoeff;          /*!< \brief Control node interpolation coefficients.*/

  unsigned long nCtrlNodesGlobal{0},      /*!< \brief Total number of control nodes.*/
  nCtrlNodesLocal{0};
  su2activematrix CtrlCoords;             /*!< \brief Coordinates of the control nodes.*/

  su2double MaxErrorGlobal{0.0};          /*!< \brief Maximum error data reduction algorithm.*/
  su2double DataReductionTolerance{0.0};
  
public:

  /*!
  * \brief Constructor of the class.
  */
  CRadialBasisFunctionInterpolation(CGeometry* geometry, CConfig* config);

  /*!
   * \brief Destructor of the class.
   */
  ~CRadialBasisFunctionInterpolation(void) override;

  /*!
   * \brief Grid deformation using the spring analogy method.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   * \param[in] UpdateGeo - Update geometry.
   * \param[in] Derivative - Compute the derivative (disabled by default). Does not actually deform the grid if enabled.
   */
  void SetVolume_Deformation(CGeometry* geometry, CConfig* config, bool UpdateGeo, bool Derivative,
                                                bool ForwardProjectionDerivative);
  
  /*!
  * \brief Selecting unique set of boundary nodes based on marker information.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem.
  */
  void SetBoundNodes(CGeometry* geometry, CConfig* config);

  /*!
  * \brief Selecting internal nodes for the volumetric deformation.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem. 
  * \param[in] internalNode - Internal nodes.
  */
  void SetInternalNodes(CGeometry* geometry, CConfig* config, vector<unsigned long>& internalNodes);

  /*!
  * \brief Assigning the control nodes.
  * \param[in] config -Definition of the particular problem.
  * */

  void SetCtrlNodes(CConfig* config);

  /*!
  * \brief Solving the RBF system to obtain the interpolation coefficients.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem.
  * \param[in] type - Type of radial basis function.
  * \param[in] radius - Support radius of the radial basis function.
  */

  void SolveRBF_System(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius);

  /*!
  * \brief Obtaining the interpolation coefficients of the control nodes.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem.
  * \param[in] type - Type of radial basis function.
  * \param[in] radius - Support radius of the radial basis function.
  */

  void GetInterpCoeffs(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius);


  /*!
  * \brief Gathering of all control node coordinates.
  * \param[in] geometry - Geometrical definition of the problem.
  */
  void SetCtrlNodeCoords(CGeometry* geometry);

  /*!
  * \brief Build the deformation vector with surface displacements of the control nodes.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem.
  */
  void SetDeformation(CGeometry* geometry, CConfig* config);

  /*!
  * \brief Computation of the interpolation matrix and inverting in.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] type - Type of radial basis function.
  * \param[in] radius - Support radius of the radial basis function.
  * \param[in] invInterpMat - Inverse of the interpolation matrix.
  */
  void ComputeInterpolationMatrix(CGeometry* geometry, const RADIAL_BASIS& type, const su2double radius, su2passivematrix& invInterpMat);

  /*!
  * \brief Computation of interpolation coefficients
  * \param[in] invInterpMat - Inverse of interpolation matrix
  */
  void ComputeInterpCoeffs(su2passivematrix& invInterpMat);

  /*!
  * \brief Finding initial data reduction control nodes based on maximum deformation.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem.
  * \param[in] maxErrorNodeLocal - Local maximum error node.
  * \param[in] maxErrorLocal - Local maximum error.
  */
  void GetInitMaxErrorNode(CGeometry* geometry, CConfig* config, vector<CRadialBasisFunctionNode*>& maxErrorNodes, su2double& maxErrorLocal);

  /*! 
  * \brief Addition of control node to the reduced set.
  * \param[in] maxErrorNode - Node with maximum error to be added.
  */
  void AddControlNode( vector<CRadialBasisFunctionNode*>& maxErrorNodes);

  /*! 
  * \brief Compute global number of control nodes.
  */
  void Get_nCtrlNodesGlobal();

  /*! 
  * \brief Compute interpolation error.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem.
  * \param[in] type - Type of radial basis function.
  * \param[in] radius - Support radius of the radial basis function.
  * \param[in] maxErrorNodeLocal - Local maximum error node.
  * \param[in] maxErrorLocal - Local maximum error.
  */
  void GetInterpError(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius,  su2double& maxErrorLocal, vector<CRadialBasisFunctionNode*>& maxErrorNodes);

  /*! 
  * \brief Compute error of single node.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem.
  * \param[in] type - Type of radial basis function.
  * \param[in] radius - Support radius of the radial basis function.
  * \param[in] iNode - Local node in consideration.
  * \param[in] localError - Local error.
  */
  void GetNodalError(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, unsigned long iNode, su2double* localError);

  /*!
  * \brief Updating the grid coordinates.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem.
  * \param[in] type - Type of radial basis function.
  * \param[in] radius - Support radius of the radial basis function.
  * \param[in] internalNodes - Internal nodes.
  */
  void UpdateGridCoord(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, const vector<unsigned long>& internalNodes);

  /*!
  * \brief Updating the internal node coordinates.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] type - Type of radial basis function.
  * \param[in] radius - Support radius of the radial basis function.
  * \param[in] internalNodes - Internal nodes.
  */
  void UpdateInternalCoords(CGeometry* geometry, const RADIAL_BASIS& type, const su2double radius, const vector<unsigned long>& internalNodes);

  /*!
  * \brief Updating the internal node coordinates.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem.
  * \param[in] type - Type of radial basis function.
  * \param[in] radius - Support radius of the radial basis function.
  */
  void UpdateBoundCoords(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius);

  /*! 
  * \brief Apply correction to the nonzero error boundary nodes.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] config - Definition of the particular problem.
  * \param[in] type - Type of radial basis function.
  * \param[in] internalNodes - Internal nodes.
  */
  void SetCorrection(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const vector<unsigned long>& internalNodes);

  /*!
  * \brief Custom comparison function, for sorting the CRadialBasisFunctionNode objects based on their index.
  * \param[in] a - First considered Radial Basis Function Node.
  * \param[in] b - Second considered Radial Basis Function Node.
  * \return True if index of a is smaller than index of b.
  */
  inline static bool HasSmallerIndex(CRadialBasisFunctionNode* a, CRadialBasisFunctionNode* b){
    return a->GetIndex() < b->GetIndex();
  }

  /*!
  * \brief Custom equality function, for obtaining a unique set of CRadialBasisFunctionNode objects.
  * \param[in] a - First considered Radial Basis Function Node.
  * \param[in] b - Second considered Radial Basis Function Node.
  * \return True if index of a and b are equal.
  */
  inline static bool HasEqualIndex(CRadialBasisFunctionNode* a, CRadialBasisFunctionNode* b){
    return b->GetIndex() == a->GetIndex();
  }

  void GetDoubleEdgeNode(const su2double* maxError, vector<CRadialBasisFunctionNode*>& maxErrorNodes);
  void CompareError(su2double* error, CRadialBasisFunctionNode* iNode, su2double& maxError, CRadialBasisFunctionNode** idx);

  void GetSlidingDeformation(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, CADTPointsOnlyClass& BoundADT);
  // CADTPointsOnlyClass GetADT(CGeometry* geometry);
  void GetSlidingNodalError(CGeometry* geometry, CConfig* config, const RADIAL_BASIS& type, const su2double radius, unsigned long iNode, su2double* localError, CADTPointsOnlyClass& BoundADT);
};