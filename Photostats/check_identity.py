from  Photostats.geometry_functions import numpy_geom
from  Photostats.geometry_functions import align_and_rmsd_numpy
from  Photostats.geometry_functions import REFLECTIONS
from  Photostats.geometry_functions import reflect_axis




def check_identity(X, Y, use_reflections = True, threshold = 0.05): 
    X_geom = numpy_geom(X)
    Y_geom = numpy_geom(Y)
    identical = False

    matches = Y.GetSubstructMatches(X, uniquify=False)
    rmsds = []
    for match in matches:
        #print(match)
        #print(Y_geom)
        swapped = Y_geom[list(match)]  
        #print(swapped)


            
        if use_reflections == True: 
            for reflection in REFLECTIONS: 
                swapped_reflection = reflect_axis(swapped, reflection)
                rmsd, aligned_coords = align_and_rmsd_numpy(X_geom, swapped_reflection, return_aligned=True)
                rmsds.append((rmsd, aligned_coords))


        else:


           rmsd, aligned_coords = align_and_rmsd_numpy(X_geom, swapped, return_aligned=True)
           rmsds.append((rmsd, aligned_coords))

    best_rmsd, best_aligned = min(rmsds, key=lambda t: t[0])
    #print(best_rmsd)
    if best_rmsd < threshold :
        identical = True

    return identical, best_rmsd