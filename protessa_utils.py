import numpy as np
from scipy.spatial import distance
from scipy.spatial import Delaunay
from ripser import ripser

def consecutiveRanges(a):
    n = len(a)
    length = 1
    l = []
    if (n == 0):
        return l
    for i in range (1, n + 1):
        if (i == n or a[i] - a[i - 1] != 1):
            if (length < 3):
                pass
            else:
                l.append(tuple([a[i - length],a[i - 1]]))
            length = 1
        else:
            length += 1
    return l

def distance_transformation(dist):
    newdist = 0
    if dist == 1:
        newdist = 0
    elif dist == 2:
        newdist = 1
    else:
        newdist = 2
    return newdist

def aspect_ratio_edge_ratio_volume(tetra):
    pa = tetra[0]
    pb = tetra[1]
    pc = tetra[2]
    pd = tetra[3]

    ad = distance.euclidean(pa,pd)
    ac = distance.euclidean(pa,pc)
    ab = distance.euclidean(pa,pb)
    bc = distance.euclidean(pb,pc)
    cd = distance.euclidean(pc,pd)
    bd = distance.euclidean(pb,pd)

    #print(ab,ac,ad,cd,bd,bc)

    E = max([ab,ac,ad,cd,bd,bc])
    e = min([ab,ac,ad,cd,bd,bc])
    V = (1/6)*abs(np.dot( np.subtract(pa,pd) , np.cross(np.subtract(pb,pd),np.subtract(pc,pd)) ))

    R = np.sqrt( (ab*cd + ac*bd + ad*bc)*(ab*cd + ac*bd - ad*bc)*(ab*cd - ac*bd + ad*bc)*(-ab*cd + ac*bd + ad*bc) )
    R = R / (24*V)

    return R/E,E/e,V

def generate_minifolds(coordinates,cutoff):
    # # create minifold data with no cutoff
    minifold = {}
    tess = Delaunay(coordinates)
    cutoff_tess = []
    # organize simplices into mini-folds
    for simplex in tess.simplices:
        simplex.sort()
        simp = [coordinates[e] for e in simplex]
        longest_edge = np.max(distance.cdist(simp,simp,metric='euclidean'))
        if longest_edge < cutoff:
            cutoff_tess.append(simplex)
            for vertex in simplex:
                if vertex in minifold.keys():
                    minifold[vertex] = np.append(np.reshape(simplex, (1, 4)),minifold[vertex],axis=0)
                else:
                    minifold[vertex] = np.reshape(simplex, (1, 4))
    return minifold, cutoff_tess

def generate_simplex_features(coordinates,minifold):
    # go through each minifold in protein
    matrix = np.zeros((len(coordinates),27))
    for i in range(len(minifold)):
        if i in minifold.keys():
            mini = minifold[i]
            # go through each simplex and apply transformation function
            for simplex in mini:
                d1 = distance_transformation(simplex[1] - simplex[0])
                d2 = distance_transformation(simplex[2] - simplex[1])
                d3 = distance_transformation(simplex[3] - simplex[2])
                # 27 features
                matrix[i][(d1*9)+(d2*3)+d3] += 1
    return matrix

def generate_persistent_homology_features(coordinates,minifold):
    hom_matrix = np.zeros((len(coordinates),6))
    # go through each minifold in protein
    for i in range(len(minifold)):
        if i in minifold.keys():
            mini = minifold[i]
            points = np.unique(mini)
            # find consecutive ranges
            consecutives = consecutiveRanges(points)
            normalized_points = []
            central_range = []
            for pair in consecutives:
                if i in np.arange(pair[0],pair[1]+1):
                    central_range = [pair[0],pair[1]]
                    break
            if len(central_range) == 0:
                hom_matrix[i][0] = 0
                hom_matrix[i][1] = 0
            else:
                for ii in range(central_range[1]-central_range[0]):
                    normalized_points.append(np.subtract(coordinates[central_range[0]+ii+1],coordinates[central_range[0]+ii]))
                normalized_points = np.asarray(normalized_points)

                r = ripser(normalized_points)
                dgms = r['dgms']
                # compute average life of n-d hole
                avg_life_h0 = sum([e[1]-e[0] for e in dgms[0][:-1]])/len(dgms[0][:-1])
                max_life_h0 = max([e[1]-e[0] for e in dgms[0][:-1]])
                if len(dgms[1]) > 0:
                    avg_life_h1 = sum([e[1]-e[0] for e in dgms[1]])/len(dgms[1])
                    max_life_h1 = max([e[1]-e[0] for e in dgms[1]])
                else:
                    avg_life_h1 = 0
                    max_life_h1 =0
                hom_matrix[i][0] = len(dgms[0])
                hom_matrix[i][1] = len(dgms[1])
                hom_matrix[i][2] = avg_life_h0
                hom_matrix[i][3] = avg_life_h1
                hom_matrix[i][4] = max_life_h0
                hom_matrix[i][5] = max_life_h1
    return hom_matrix

def generate_dihedral_edgeratio_features(coordinates,minifold):
    matrix = np.zeros((len(coordinates),2))
    # go through each minifold in protein
    for i in range(len(minifold)):
        if i in minifold.keys():
            mini = minifold[i]
            points = np.unique(mini)
            # find consecutive ranges
            consecutives = consecutiveRanges(points)
            normalized_points = []
            central_range = []
            for pair in consecutives:
                if i in np.arange(pair[0],pair[1]+1):
                    central_range = [pair[0],pair[1]]
                    break
            omegas = []
            ers = []
            if len(central_range)>0 and central_range[1] - central_range[0]+ 1> 3:
                for ii in range(central_range[0],central_range[1]-2):
                    u1 = np.subtract(coordinates[ii+1],coordinates[ii])
                    u2 = np.subtract(coordinates[ii+2],coordinates[ii+1])
                    u3 = np.subtract(coordinates[ii+3],coordinates[ii+2])
                    n = np.dot(np.cross(u1,u2),np.cross(u2,u3))
                    d = np.linalg.norm(np.cross(u1,u2))*np.linalg.norm(np.cross(u2,u3))
                    omega = np.arccos(n/d)
                    omegas.append(omega)

                    ar,er,v = aspect_ratio_edge_ratio_volume([coordinates[ii] for ii in [ii,ii+1,ii+2,ii+3]])
                    ers.append(er)
            avg_omega = 0
            avg_er = 0
            if len(omegas) > 0:
                avg_omega = np.mean(omegas)
                avg_er = np.mean(ers)
            matrix[i][0] = avg_omega
            matrix[i][1] = avg_er
    return matrix

def generate_edgeratio_features(coordinates,minifold):
    matrix = np.zeros((len(coordinates),2))
    # go through each minifold in protein
    for i in range(len(minifold)):
        if i in minifold.keys():
            mini = minifold[i]
            points = np.unique(mini)
            # find consecutive ranges
            consecutives = consecutiveRanges(points)
            normalized_points = []
            central_range = []
            for pair in consecutives:
                if i in np.arange(pair[0],pair[1]+1):
                    central_range = [pair[0],pair[1]]
                    break
            ers = []
            central_range_simps_ers = []
            avg_er = 0
            avg_cr_er = 0
            if len(central_range)>0 and central_range[1] - central_range[0]+ 1> 3:
                for ii in range(central_range[0],central_range[1]-2):
                    ar,er,v = aspect_ratio_edge_ratio_volume([coordinates[ii] for ii in [ii,ii+1,ii+2,ii+3]])
                    ers.append(er)
                central_range_residues = [e for e in range(central_range[0],central_range[1]+1)]
                central_range_simps = []
                for simp in mini:
                    v_count = 0
                    for v in simp:
                        if v in central_range_residues:
                            v_count +=1
                        else:
                            pass
                    if v_count > 1:
                        central_range_simps.append(list(simp))
                for simp in central_range_simps:
                    ar,er,v = aspect_ratio_edge_ratio_volume([coordinates[ii] for ii in simp])
                    central_range_simps_ers.append(er)
            if len(central_range_simps_ers) > 0:
                avg_cr_er = np.mean(central_range_simps_ers)
            if len(ers) > 0:
                avg_er = np.mean(ers)
            matrix[i][0] = avg_er
            matrix[i][1] = avg_cr_er
    return matrix
