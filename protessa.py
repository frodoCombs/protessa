import argparse
import sys
import numpy as np
from protessa_utils import *
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
__author__ = "Ford Combs"
__version__ = "0.1.0"
__license__ = "MIT"
def parse_chains(file_path):
    chains = {}
    no_complete_chains = True
    with open(file_path,'r',encoding='utf-8') as f:
        aminos = {}
        atoms = {}
        for line in f:
            if 'ATOM' in line[0:7] or 'HETATM' in line[0:7]:
                # print(line)
                chain = line[21]
                position = int(line[22:26].replace(' ',''))
                atom = line[13:16].replace(' ','')
                residue = line[17:20].replace(' ','')
                if atom == 'CA':
                    x = float(line[30:38].replace(' ',''))
                    y = float(line[38:46].replace(' ',''))
                    z = float(line[46:54].replace(' ',''))
                    if chain in chains.keys():
                        if position not in chains[chain]['positions']:
                            chains[chain]['positions'].append(position)
                    else:
                        chains[chain] = {}
                        chains[chain]['positions'] = [position]

        # check that all residues are consecutive
        for chain in chains.keys():
            chains[chain]['simplified positions'] = [chains[chain]['positions'][i]-chains[chain]['positions'][i-1] for i in range(1,len(chains[chain]['positions']))]
            chains[chain]['data'] = 'Length of chain ' + str(len(chains[chain]['positions']))
            unique_positions = np.unique([chains[chain]['positions'][i]-chains[chain]['positions'][i-1] for i in range(1,len(chains[chain]['positions']))])
            if len(unique_positions) == 1 and unique_positions[0] == 1:
                chains[chain]['data'] += ' and no missing alpha carbons'
                chains[chain]['complete'] = True
                no_complete_chains = False
            else:
                chains[chain]['complete'] = False
                chains[chain]['data'] += ' segments are '
                segments = [chains[chain]['positions'][0]]
                for i in range(1,len(chains[chain]['positions'])):
                    if chains[chain]['positions'][i] - chains[chain]['positions'][i-1] != 1:
                        segments.append(chains[chain]['positions'][i-1])
                        segments.append(chains[chain]['positions'][i])
                    elif i == len(chains[chain]['positions'])-1:
                        segments.append(chains[chain]['positions'][i])
                    else:
                        pass
                chains[chain]['segments'] = segments
                chains[chain]['data'] += ','.join([str(e) for e in segments])
    return chains, no_complete_chains
# function to parse pdb file
def parse_pdb(file_path, selected_chain):
    with open(file_path,'r',encoding='utf-8') as f:
        aminos = {}
        atoms = {}
        for line in f:
            if 'ATOM' in line[0:7] or 'HETATM' in line[0:7]:
                # print(line)
                chain = line[21]
                position = int(line[22:26].replace(' ',''))
                atom = line[13:16].replace(' ','')
                residue = line[17:20].replace(' ','')
                if atom == 'CA' and chain  == selected_chain:
                    x = float(line[30:38].replace(' ',''))
                    y = float(line[38:46].replace(' ',''))
                    z = float(line[46:54].replace(' ',''))
                    # print(chain,position,atom,residue,x,y,z)
                    if position in atoms.keys():
                        atoms[position].append([atom,x,y,z])
                    else:
                        atoms[position] = [[atom,x,y,z]]
                        aminos[position] = [residue]
        # check that all residues are consecutive
        positions = list(atoms.keys())
        positions = np.unique([positions[i]-positions[i-1] for i in range(1,len(positions))])
        if len(positions) == 1 and positions[0] == 1:
            # proteins[key] = [atoms,aminos]
            return atoms,aminos
        else:
            return 0,0

def perform_ssa(file_path,model,chain,cutoff=8):
    # load model
    if model == 'auth':
        # Load RF model
        rf = pkl.load(open("27features_ER_PH_author.pkl",'rb'))
        ss2int = {'H': 0, 'E': 1, 'C': 2}
    elif model == 'dssp':
        # Load RF model
        rf = pkl.load(open("27features_ER_PH_dssp.pkl",'rb'))
        ss2int = {'H': 0, 'E': 1, 'C': 2}
    elif model == 'stride':
        # Load RF model
        rf = pkl.load(open("27features_ER_PH_stride.pkl",'rb'))
        ss2int = {'H': 0, 'E': 1, 'C': 2}
    else:
        model = 'kmeans'
        # Load K-means models
        km = pkl.load(open("27features_ER_PH_Kmeansk3_cluster.pkl",'rb'))
    # set tessellation cutoff
    if cutoff == '8':
        cutoff = 8
    elif cutoff == '10':
        cutoff = 10
    else:
        # no cutoff, arbitrary high value
        cutoff = 10000
    atoms,aminos = parse_pdb(file_path,chain)
    alpha_carbons = []
    for i in list(atoms.keys()):
        for atom in atoms[i]:
            if atom[0] == 'CA':
                alpha_carbons.append(atom[1:])
    minifold, tess = generate_minifolds(alpha_carbons,cutoff)
    simp_features = generate_simplex_features(alpha_carbons,minifold)
    pm_features = generate_persistent_homology_features(alpha_carbons,minifold)
    er_features = generate_edgeratio_features(alpha_carbons,minifold)
    protein_features = np.append(simp_features,pm_features,1)
    protein_features = np.append(protein_features,er_features,1)

    if model != 'kmeans':
        # One model system
        predictions = rf.predict(protein_features)
    else:
        km2ss = {2:'C',0:'E',1:'H'}
        predictions = [km2ss[e] for e in km.predict(protein_features)]
    print(''.join(predictions))

def main(args):
    if args.chain == "none":
        chains,no_complete_chains = parse_chains(args.pdb_file)
        if no_complete_chains == True:
            print('This protein contains no complete chains')
        else:
            print('Please choose a chain from the list below:')
            print('Complete chain(s): ',end=' ')
            for chain in chains:
                if chains[chain]['complete'] == True:
                    print(chain,end=' ')
                print()
    else:
        perform_ssa(args.pdb_file,args.model,args.chain,args.edge_length_cutoff)
    # print(chains,no_complete_chains)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Optional argument flag which defaults to False
    parser.add_argument("-c", "--chain", action="store", default="none",help="Chain ID {A,B,C,etc.}")
    parser.add_argument("-m", "--model", action="store", default="none",\
    help="Please select the ProTeSSA model you would like to use from: {auth,dssp,stride,or kmeans}\
    auth = model trained on PDB structures author(s)' SSAs;\
    dssp = model trained on DSSP SSAs;\
    stride = model trained on STRIDE SSAs;\
    kmeans = k-means (k=3) cluster model ")
    parser.add_argument("-e", "--edge-length-cutoff", action="store", default=8,help="Edge length cutoff {8,10,etc.}")
    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-f", "--pdb-file", action="store")
    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="ProTeSSA version {version}".format(version=__version__))
    args = parser.parse_args()
    main(args)
