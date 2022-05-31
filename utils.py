import os
import logging
import re
from types import SimpleNamespace

import torch
import torch.distributed as dist
import cv2
import numpy as np


def get_slurm_nodelist():                                                       
    """Gets SLURM_NODELIST"""                                                   
    try:                                                                        
        return os.environ['SLURM_NODELIST']                                     
    except KeyError:                                                            
        raise RuntimeError('SLURM_NODELIST not found in environment')


def expand_hostlist(hostlist):                                                  
    """Create a list of hosts from hostlist"""                                  
                                                                                
    def split_hostlist(hostlist):                                               
        """Split hostlist as commas outside of range expressions ('[3-5]')"""   
        in_brackets = False                                                     
        cur_host = ''                                                           
        for c in hostlist:                                                      
            if in_brackets:                                                     
                assert c != '['                                                 
                if c == ']':                                                    
                    in_brackets = False                                         
            elif c == '[':                                                      
                in_brackets = True                                              
            elif c == ',':                                                      
                assert cur_host != ''                                           
                yield cur_host                                                  
                cur_host = ''                                                   
                continue                                                        
            cur_host += c                                                       
        if cur_host:                                                            
            yield cur_host                                                      
                                                                                
    def expand_range_expression(range_exp):                                     
        """Expand a range expression like '3-5' to 3,4,5"""                     
        for part in range_exp.split(','):                                       
            sub_range = part.split('-')                                         
            if len(sub_range) == 1:                                             
                sub_range = sub_range * 2                                       
            else:                                                               
                assert len(sub_range) == 2                                      
            for i in range(int(sub_range[0]), int(sub_range[1]) + 1):           
                yield i

    hosts = []                                                                  
    try:                                                                        
        for part in split_hostlist(hostlist):                                   
            m = re.match(r'([^,[\]]*)(\[([^\]]+)\])?$', part)                   
            if m is None:                                                       
                raise ValueError('Invalid part: {}'.format(part))               
            prefix = m.group(1) or ''                                           
            if m.group(3) is None:                                              
                hosts.append(prefix)                                            
            else:                                                               
                hosts.extend(                                                   
                    prefix + str(i) for i in expand_range_expression(m.group(3))
                )                                                               
    except Exception as e:                                                      
        raise ValueError('Invalid hostlist format "{}": {}'.format(hostlist, e))
    return hosts 


def setup(args):                                                    
    """Setup process group"""                                                   
    if args.dist == 0:                                                 
        try:                                                                    
            rank = int(os.environ['RANK'])                                      
            size = int(os.environ['WORLD_SIZE'])                                
        except KeyError:                                                        
            raise Exception('fastai.launch variables are not set')              

    elif args.dist == 1:                                                      
        try:                                                                    
           rank = int(os.environ['OMPI_COMM_WORLD_RANK'])                       
           size = int(os.environ['OMPI_COMM_WORLD_SIZE'])                       
        except KeyError:                                                        
            raise Exception('OpenMPI variables are not set')                    
                                                                                
        os.environ['RANK'] = str(rank)                                          
        os.environ['WORLD_SIZE'] = str(size)                                    

    elif args.dist == 2:
        rank = 0
        size = 1
    elif args.dist == 3:
        try:
            rank = int(os.environ['SLURM_PROCID'])
            size = int(os.environ['SLURM_NTASKS'])
        except KeyError:
            raise Exception('SLURM variables are not set')
    else:                                                                       
        raise Exception('args.dist ({}) is invalid'.          
            format(args.dist)                                                   
        )                                                                       
                                                                                
                                                                                
    # Assumes that all nodes have same number of GPUs                           
    local_size = torch.cuda.device_count()                                      
    local_rank = rank % local_size

    # Uses SLURM environment variables to get master node
    nodelist = get_slurm_nodelist()                                             
    hosts = expand_hostlist(nodelist)                                           
    master = hosts[0]                                                           
    master = '{}.dug.com'.format(master)
                                                                                
    #PyTorch required environment variables
    os.environ['MASTER_ADDR'] = master                                          
    os.environ['MASTER_PORT'] = '24355' # random port                                       
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)
                                                                 
    dist.init_process_group('nccl', rank=rank, world_size=size)

    logging.info('rank: {}\nsize: {}\nlocal_rank: {}\nlocal_size: {}\n'.
            format(rank, size, local_rank, local_size))

    return SimpleNamespace(
        rank=rank,
        size=size,
        local_rank=local_rank,
        local_size=local_size
    )


def cleanup():
    """Destroys process group"""
    dist.destroy_process_group()
