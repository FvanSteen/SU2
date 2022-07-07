#!/usr/bin/env python 

## \file amg.py
#  \brief python script for running mesh adaptation using the AMG Inria library
#  \author Victorien Menier, Brian Mungu\'ia
#  \version 7.3.0 "Blackbird"
#
# SU2 Project Website: https://su2code.github.io
# 
# The SU2 Project is maintained by the SU2 Foundation 
# (http://su2foundation.org)
#
# Copyright 2012-2022, SU2 Contributors (cf. AUTHORS.md)
#
# SU2 is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# SU2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with SU2. If not, see <http://www.gnu.org/licenses/>.

import os, shutil, copy, time

from .. import io   as su2io
from .. import amginria as su2amg
from .interface import CFD as SU2_CFD

def amg(config):
    """
    Runs the a mesh adaptation loop with the AMG library.
        
    Inputs:
        config - an SU2 config object
    """
    
    print('SU2-AMG Anisotropic Mesh Adaptation')
        
    #--- Check config options related to mesh adaptation
    
    pyadap_options = [ 'ADAP_SIZES', 'ADAP_SUBITER', 'ADAP_SENSOR', 'ADAP_BACK', \
                       'ADAP_HGRAD', 'ADAP_RESIDUAL_REDUCTION', 'ADAP_FLOW_ITER', \
                       'ADAP_ADJ_ITER', 'ADAP_CFL', 'ADAP_INV_BACK', 'ADAP_ORTHO', \
                       'ADAP_RDG' ]
    required_options = [ 'ADAP_SIZES', 'ADAP_SUBITER', 'ADAP_SENSOR', 'ADAP_HMAX', \
                         'ADAP_HMIN', 'MESH_FILENAME', 'RESTART_SOL', 'MESH_OUT_FILENAME' ]
    
    if not all (opt in config for opt in required_options):
        err = '\n\n## ERROR : Missing options: \n'
        for opt in required_options:
            if not opt in config:
                err += opt + '\n'
        raise AttributeError(err)
    
    #--- Print adap options

    print(su2amg.print_adap_options(config))
    
    #--- How many iterative loops? Using what prescribed mesh sizes? 
    
    mesh_sizes   = su2amg.get_mesh_sizes(config)
    sub_iter     = su2amg.get_sub_iterations(config)
    
    #--- Solver iterations/ residual reduction param for each size level

    adap_flow_iter = su2amg.get_flow_iter(config)
    adap_adj_iter  = su2amg.get_adj_iter(config)
    adap_flow_cfl  = su2amg.get_flow_cfl(config)

    adap_sensor = config.ADAP_SENSOR
    sensor_avail = ['MACH', 'PRES', 'MACH_PRES', 'GOAL']
    
    if adap_sensor not in sensor_avail:
        raise ValueError(f'Unknown adaptation sensor {adap_sensor}. Available options are {sensor_avail}.')
        
    if len(mesh_sizes) != len(sub_iter):
        raise ValueError(f'Inconsistent number of mesh sizes and sub-iterations. {len(mesh_sizes)} mesh sizes and {len(sub_iter)} sub-iterations provided.')
    
    #--- Change current directory
    
    warn = True
    adap_dir = './adap'
    cwd = os.getcwd()
        
    if os.path.exists(adap_dir):
        print('./adap exists. Removing old mesh adaptation in 10s.')
        if warn : time.sleep(10)
        shutil.rmtree(adap_dir)
        print(f'The {adap_dir} folder was deleted.')
    
    os.makedirs(adap_dir)
    os.chdir(adap_dir)
    
    cur_dir = './ite0'
    os.makedirs(cur_dir)
    os.chdir(cur_dir)
    os.symlink(os.path.join(cwd, config.MESH_FILENAME), config.MESH_FILENAME)
        
    cur_meshfil = config['MESH_FILENAME']

    #--- Format of history file

    history_format = config.TABULAR_FORMAT
    if (history_format == 'TECPLOT'):
        history_filename = os.path.join(cwd,'history_adap.dat')
    else:
        history_filename = os.path.join(cwd,'history_adap.csv')

    #--- Get mesh dimension

    dim = su2amg.get_su2_dim(cur_meshfil)
    if ( dim != 2 and dim != 3 ):
        raise ValueError('Wrong dimension number.')
    
    #--- AMG parameters

    config_amg = su2amg.get_amg_config(config)

    #--- Compute initial solution if needed, else link current files
    
    config_cfd = copy.deepcopy(config)
    config_cfd_ad = copy.deepcopy(config)
    for opt in pyadap_options:
        config_cfd.pop(opt, None)
        config_cfd_ad.pop(opt, None)

    #--- Check config for filenames if restarting
    if config['RESTART_SOL'] == 'YES':
        required_options=['SOLUTION_FILENAME','SOLUTION_ADJ_FILENAME']
        if not all (opt in config for opt in required_options):
            err = 'RESTART_SOL is set to YES, but the solution is missing:\n'
            for opt in required_options:
                if not opt in config:
                    err += opt + '\n'
            raise ValueError(err)

        os.symlink(os.path.join(cwd, config.SOLUTION_FILENAME), config.SOLUTION_FILENAME)

        print('Initial CFD solution is provided.')

    else:
        print('Running initial CFD solution.')

    #--- Only allow binary restarts since WRT_BINARY_RESTART is deprecated
    sol_ext = '.dat'

    cur_meshfil = config['MESH_FILENAME']
    cur_solfil  = f'restart_flow{sol_ext}'
    su2amg.set_flow_config_ini(config_cfd, cur_solfil)

    try: # run with redirected outputs
        #--- Run a single iteration of the flow if restarting to get history info
        if config['RESTART_SOL'] == 'YES':
            config_cfd.ITER = 1
            config_cfd.RESTART_CFL = 'YES'

        with su2io.redirect.output('su2.out'): SU2_CFD(config_cfd)

        if config['RESTART_SOL'] == 'YES':
            os.remove(cur_solfil)
            os.symlink(os.path.join(cwd, config.SOLUTION_FILENAME), cur_solfil)

        #--- Set RESTART_SOL=YES for runs after adaptation
        config_cfd.RESTART_SOL = 'YES'
        config_cfd.RESTART_CFL = 'YES'

        if adap_sensor == 'GOAL':
            cur_solfil_adj = f'restart_adj{sol_ext}'
            su2amg.set_adj_config_ini(config_cfd_ad, cur_solfil, cur_solfil_adj, mesh_sizes[0])

            #--- If restarting, check for the existence of an adjoint restart
            if config['RESTART_SOL'] == 'YES':
                cur_solfil_adj_ini = config_cfd_ad.SOLUTION_ADJ_FILENAME    
                func_name          = config.OBJECTIVE_FUNCTION
                suffix             = su2io.get_adjointSuffix(func_name)
                cur_solfil_adj_ini = su2io.add_suffix(cur_solfil_adj_ini,suffix)

                #--- Run an adjoint if the solution file doesn't exist
                if not (os.path.exists(os.path.join(cwd, cur_solfil_adj_ini))):
                    config_cfd_ad.ITER        = config.ITER
                    config_cfd_ad.RESTART_SOL = 'NO'

                    print('Running initial adjoint CFD solution.')

                #--- Otherwise just compute the metric
                else:
                    os.symlink(os.path.join(cwd, cur_solfil_adj_ini), cur_solfil_adj_ini)
                    config_cfd_ad.ITER = 0

                    print('Initial adjoint CFD solution is provided.')

            else:
                print('Running initial adjoint CFD solution.')

            with su2io.redirect.output('su2.out'): SU2_CFD(config_cfd_ad)

            func_name      = config.OBJECTIVE_FUNCTION
            suffix         = su2io.get_adjointSuffix(func_name)
            cur_solfil_adj = su2io.add_suffix(cur_solfil_adj,suffix)

            #--- Set RESTART_SOL=YES for runs after adaptation
            config_cfd_ad.RESTART_SOL = 'YES'

    except:
        raise
        
    #--- Check existence of initial mesh, solution
    
    required_files = [cur_meshfil,cur_solfil]
    
    if not all (os.path.exists(fil) for fil in required_files):
        err = "Can't find the following files:\n"
        for fil in required_files:
            if not os.path.exists(fil):
                err += fil + '\n'
        raise Exception(err)
    
    #--- Start adaptive loop

    global_iter = 0

    #--- Print convergence history

    npoin = su2amg.get_su2_npoin(cur_meshfil)
    su2amg.plot_results(history_format, history_filename, global_iter, npoin)
    
    print('\nStarting mesh adaptation process.')
    
    for iSiz in range(len(mesh_sizes)):
        
        mesh_size = int(mesh_sizes[iSiz])
        nSub      = int(sub_iter[iSiz])
                        
        print(f'\nIteration {iSiz} - Mesh size coefficient {mesh_size}')
        
        for iSub in range(nSub):

            global_iter += 1
            
            # Prints
            pad_cpt = f'({iSub+1}/{nSub})'.ljust(9)
            pad_nul = ''.ljust(9)
            
            #--- Load su2 mesh 
            
            mesh = su2amg.read_mesh_and_sol(cur_meshfil, cur_solfil)

            #--- Write solution
            su2amg.write_mesh_and_sol('flo.meshb', 'flo.solb', mesh)

            config_amg['size'] = mesh_size
                
            #--- Use pyAmg interface
            
            if adap_sensor == 'GOAL':

                #--- Use metric computed from SU2 to drive the adaptation

                metric_wrap = su2amg.create_sensor(mesh, adap_sensor)
                mesh['metric'] = metric_wrap['solution']

                #--- Read and merge adjoint solution to be interpolated

                sol_adj = su2amg.read_sol(cur_solfil_adj, mesh)
                su2amg.merge_sol(mesh, sol_adj)

                del sol_adj

            else:        

                #--- Create sensor used to drive the adaptation  

                sensor_wrap = su2amg.create_sensor(mesh, adap_sensor)
                mesh['sensor'] = sensor_wrap['solution']
                
            #--- Call pyAMG

            print(f' {pad_cpt} Generating adapted mesh using AMG')
            
            mesh_new = su2amg.call_pyamg(mesh, config_amg)

            #--- Remove extra files generated by AMG

            extra_files=['back.meshb','meshp3_smoo.meshb','optim.0.meshb','optim.0.solb','subdom.meshb']
            for file in extra_files:
                try:
                    os.remove(file)
                except OSError:
                    pass
                            
            #--- print mesh size
            
            print(f' {pad_nul} AMG done: {su2amg.get_mesh_size(mesh_new)}')

            mesh_new['markers'] = mesh['markers']
            mesh_new['dimension'] = mesh['dimension']
            mesh_new['solution_tag'] = mesh['solution_tag']

            del mesh

            old_dir = cur_dir
            cur_dir = f'./ite{global_iter}'
            os.makedirs(os.path.join('..',cur_dir))
            os.chdir(os.path.join('..',cur_dir))
            
            cur_meshfil = 'adap.su2'
            cur_solfil  = f'flo{sol_ext}'
                            
            su2amg.write_mesh_and_sol(cur_meshfil, cur_solfil, mesh_new)

            if adap_sensor == 'GOAL':
                cur_solfil_adj = f'adj{sol_ext}'
                sol_adj = su2amg.split_adj_sol(mesh_new)
                su2amg.write_sol(cur_solfil_adj, sol_adj)

            cur_meshfil_gmf    = 'flo_itp.meshb'
            cur_solfil_gmf     = 'flo_itp.solb'
            su2amg.write_mesh_and_sol(cur_meshfil_gmf, cur_solfil_gmf, mesh_new)

            del mesh_new

            if adap_sensor == 'GOAL':
                cur_solfil_gmf_adj = 'adj_itp.solb'
                su2amg.write_sol(cur_solfil_gmf_adj, sol_adj)
                del sol_adj
                
            #--- Run su2
            
            print(f' {pad_nul} Running CFD')
        
            try: # run with redirected outputs
                
                cur_solfil_ini = f'flo_ini{sol_ext}'
                os.rename(cur_solfil, cur_solfil_ini)
                
                su2amg.update_flow_config(config_cfd, cur_meshfil, cur_solfil, cur_solfil_ini, \
                                          adap_flow_iter[iSiz], adap_flow_cfl[iSiz])
                
                with su2io.redirect.output('su2.out'): SU2_CFD(config_cfd)
                
                if not os.path.exists(cur_solfil) :
                    raise RuntimeError('SU2_CFD failed.\n')

                #--- Print convergence history

                npoin = su2amg.get_su2_npoin(cur_meshfil)
                su2amg.plot_results(history_format, history_filename, global_iter, npoin)
                    
                if adap_sensor == 'GOAL':

                    cur_solfil_adj_ini = f'adj_ini{sol_ext}'
                    cur_solfil_adj_ini = su2io.add_suffix(cur_solfil_adj_ini,suffix)
                    os.rename(cur_solfil_adj, cur_solfil_adj_ini)
                    cur_solfil_adj_ini = f'adj_ini{sol_ext}'

                    su2amg.update_adj_config(config_cfd_ad, cur_meshfil, cur_solfil, cur_solfil_adj, \
                                             cur_solfil_adj_ini, adap_adj_iter[iSiz], mesh_sizes[iSiz])

                    with su2io.redirect.output('su2.out'): SU2_CFD(config_cfd_ad)

                    cur_solfil_adj = su2io.add_suffix(cur_solfil_adj,suffix)

                    if not os.path.exists(cur_solfil_adj) :
                        raise RuntimeError('SU2_CFD_AD failed.\n')
            
            except:
                raise

    #--- Write final files

    mesh = su2amg.read_mesh_and_sol(cur_meshfil, cur_solfil)
    su2amg.write_mesh_and_sol('flo.meshb', 'flo.solb', mesh)
    
    os.rename(cur_solfil,os.path.join(cwd,config.RESTART_FILENAME))
    os.rename(cur_meshfil,os.path.join(cwd,config.MESH_OUT_FILENAME))
    
    print('\nMesh adaptation successfully ended. Results files:')
    print(f'{config.MESH_OUT_FILENAME}\n{config.RESTART_FILENAME}\n\n')
    
