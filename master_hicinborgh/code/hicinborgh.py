#For output and reading MUSTACHE output
import csv

#Tracking RAM
import psutil, os

#Core Functionality
import numpy as np
import skimage as ski
import cooler
from scipy.ndimage import gaussian_filter, maximum_filter, label

#Extra Mathematics
import math
from scipy.stats import norm, laplace, expon
from statsmodels.stats.multitest import multipletests
import warnings

#For Distance Enrichment Score
from collections import defaultdict

#For Binary Matrix
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

#Needed for memory
import gc

#Visualization of the graph
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

#Ease of use
from pathlib import Path

#To detect elapsed time to run
import time

#Obtains a value based on the Determinant of Hessian equation at sigma scales
#Returns the response values in a 3D Matrix [y, x, s for sigma]
def compute_hessian_response(subreg):
    
    sigmas = np.geomspace(1, 8, 12) #min_sigma, max_sigma, num_sigmas

    responses = []

    for sigma in sigmas:

        #Obtaining the Hessian Matrix of the subregion
        Hxx, Hxy, Hyy = ski.feature.hessian_matrix(
            subreg.astype(float),
            sigma=sigma,
            order='rc',
            use_gaussian_derivatives=True
        )

        #Obtaining the value of the Determinant of the Hessian Matrix
        detH = sigma**4 * (
            Hxx * Hyy -
            Hxy**2
        )

        responses.append(detH)

    response_volume = np.stack(
        responses,
        axis=-1
    )

    return response_volume

def compute_dog_response(subreg):

    sigmas = np.geomspace(1, 8, 12)

    responses = []

    for sigma in sigmas:

        # Gaussian at current scale
        G1 = gaussian_filter(
            subreg.astype(float),
            sigma=sigma
        )

        # Gaussian at next scale
        G2 = gaussian_filter(
            subreg.astype(float),
            sigma=sigma * np.sqrt(2)
        )

        # Difference of Gaussian
        dog = G1 - G2

        # Scale normalization
        dog = sigma * dog

        responses.append(dog)


    response_volume = np.stack(
        responses,
        axis=-1
    )

    return response_volume

# Returns a 3D Boolean Array that tells where the local maxima is
# in a 3x3x3 neighborhood
#
# Conditions:
# 1. valid
#   The candidate must be at least 4 diagonals away from the center diagonal
# 2. c > 0
#   The current scale maxima must be greater than 0
# 3. c == c_max,
#   The current scale blob evaluated must be the maxima in a 3x3 window
# 4. (p == p_max) | (n == n_max),
#   Either the previous or next scale of blobs must also be maxima in a 3x3 window in their own scales.
# 5. c > p_max,
#   The current scale blob must be greater than the greatest of the previous scale
# 6. c > n_max
#   The current scale blob must be greater than the greatest of the next scale
#
# If they survive these conditions, they are allowed to procede as candidate loops
def local_3d_max(response_volume):

    valid = np.triu(
        np.ones(response_volume.shape[:2], dtype=bool),
        4
    )

    maxima3d = np.zeros_like(
        response_volume,
        dtype=bool
    )

    n_scales = response_volume.shape[-1]

    vAll = np.full(
        response_volume.shape[:2],
        -np.inf
    )

    for s in range(1, n_scales-1):

        p = response_volume[:,:,s-1]
        c = response_volume[:,:,s]
        n = response_volume[:,:,s+1]

        p_max = maximum_filter(
            p,
            size=3
        )

        c_max = maximum_filter(
            c,
            size=3
        )

        n_max = maximum_filter(
            n,
            size=3
        )

        keep = np.logical_and.reduce(
            (
                valid,
                c > 0,         
                c > vAll,
                c == c_max,
                (p == p_max) | (n == n_max),
                c > p_max,
                c > n_max
            )
        )

        maxima3d[:,:,s] = keep

        vAll[keep] = c[keep]

    return maxima3d

# Creating a Laplace Distribution and getting 
# Laplace Parameters from the candidate loops
#
# In the program, we best use this for Difference of Gaussian
#
# Returns the Laplace parameters
def computing_laplace(loop_candidates):
    by_scale = defaultdict(list)

    for y, x, s, score in loop_candidates:
        by_scale[s].append(score)

    laplace_params = {}

    for s, vals in by_scale.items():
        vals = np.array(vals)

        #Obtaining the Laplace Parameters based on DoH Scores for each loop and storing them
        loc, scale = laplace.fit(vals) 
        laplace_params[s] = (loc, scale)

    return laplace_params


# Applies the Laplace Survival Function
# on the Laplace Distribution to gain p-values for each loop
#
# Returns the loops with their designated p-values
def computing_pval_l(candidates, laplace_params):
    candidate_loops = []

    for y, x, s, score in candidates:

        loc, scale = laplace_params[s]

        pval = laplace.sf(score, loc=loc, scale=scale)

        candidate_loops.append(
            (y, x, s, score, pval)
        )

    return candidate_loops

# Creating an Exponential Distribution and getting 
# Exponential Parameters from the candidate loops
#
# In the program, we best use this for Determinant of Hessian
#
# Returns the Exponential parameters 
def computing_exponential(loop_candidates):
    by_scale = defaultdict(list)

    for y, x, s, score in loop_candidates:
        by_scale[s].append(score)

    exp_params = {}

    for s, vals in by_scale.items():

        vals = np.array(vals)

        # Removes tiny negative values caused by floating point
        vals = np.clip(vals, 0, None)

        loc, scale = expon.fit(vals)

        exp_params[s] = (loc, scale)

    return exp_params

# Applies the Exponential Survival Function
# on the Exponential Distribution to gain p-values for each loop
#
# Returns the loops with their designated p-values
def computing_pval_e(candidates, exp_params):

    candidate_loops = []

    for y, x, s, score in candidates:

        loc, scale = exp_params[s]

        score = max(score, 0)

        pval = expon.sf(
            score,
            loc=loc,
            scale=scale
        )

        candidate_loops.append(
            (y, x, s, score, pval)
        )

    return candidate_loops

# Checks a 3x3 area surrounding a local maxima pixel
# for other local maxima to combine them as one loop
#
# Returns the candidate loops left after connecting them
def binary_matrix_global(cand_pval, shape):
    binary = np.zeros(shape, dtype=np.uint8)

    for y, x, s, score, pval in cand_pval:
        binary[y, x] = 1

    structure = np.ones((3, 3), dtype=int)
    labels, _ = label(binary, structure=structure)

    groups = defaultdict(list)

    for cand in cand_pval:
        y, x, s, score, pval = cand
        
        label_id = labels[y,x]

        if label_id == 0:
            continue

        groups[label_id].append(cand)

    representatives = [
        min(group, key=lambda c: c[4])
        for group in groups.values()
    ]

    return representatives


# Take a candidate loop and its sigma it was found with,
# make a square window with each side equal to its sigma,
# and check if it is as sparse as its threshold desired. 
# If it is too sparse, remove the candidate
#
# Returns the candidates that surpass the sparsity threshold
def filter_sparse(shape_y, shape_x, representatives, raw_hic_numpy, st=0.8):
    
    filtered_candidates = []

    # MUSTACHE's conditions for true loops:
    # nz = np.logical_and(c != 0, np.triu(c, 4))
    nz = np.logical_and(
        raw_hic_numpy != 0,
        np.triu(raw_hic_numpy,4)
    )

    for y, x, sigma, score, pval in representatives:

        # First window: radius = sigma
        s = math.ceil(sigma)

        r1 = max(0, y - s)
        r2 = min(shape_y, y + s + 1)

        c1_start = max(0, x - s)
        c1_end = min(shape_x, x + s + 1)

        window1 = nz[
            r1:r2,
            c1_start:c1_end
        ]

        c1 = np.sum(window1) / window1.size


        # Second window: radius = 2*sigma
        s = 2 * s

        r1 = max(0, y - s)
        r2 = min(shape_y, y + s + 1)

        c2_start = max(0, x - s)
        c2_end = min(shape_x, x + s + 1)

        window2 = nz[
            r1:r2,
            c2_start:c2_end
        ]

        c2 = np.sum(window2) / window2.size


        # MUSTACHE filtering condition
        if c1 < st or c2 < 0.6:
            continue


        filtered_candidates.append(
            (
                y,
                x,
                sigma,
                score,
                pval
            )
        )

    return filtered_candidates

# Checks the set of loops for a specific chromosome
# and calculates the APA score for the entire chromosome.
# This is what tells us how strong a loop is.
#
# Returns the APA score for a chromosome
def apa(raw_hic_numpy, final_loops): # Aggregate Peak Analysis
    window_radius = 10

    patches = []

    for chrom, y, x, sigma, score, p, q in final_loops:

        if (
            y-window_radius < 0 or
            x-window_radius < 0 or
            y+window_radius >= raw_hic_numpy.shape[0] or
            x+window_radius >= raw_hic_numpy.shape[1]
        ):
            continue

        patch = raw_hic_numpy[
            y-window_radius:y+window_radius+1,
            x-window_radius:x+window_radius+1
        ]

        patches.append(patch)

    apa_matrix = np.mean(patches, axis=0)

    center = apa_matrix[10,10]

    background = np.mean(
        apa_matrix[-5:, -5:]
    )

    apa_score = center / background

    print("APA score =", apa_score)

    return apa_score

# The exact normalization function from MUSTACHE
# Only takes sparse arguments
#
# Returns the normalized contact map as sparse arguments
def normalize_sparse(x, y, v, resolution, distance_in_px):
    n = max(max(x), max(y)) + 1

    pval_weights = []
    distances = np.abs(y - x)
    if (n - distance_in_px) * resolution > 2000000:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            filter_size = int(2000000 / resolution)
            for d in range(2 + distance_in_px):
                indices = distances == d
                vals = np.zeros(n - d)
                vals[x[indices]] = v[indices] + 0.001
                if vals.size == 0:
                    continue
                std = np.std(v[indices])
                mean = np.mean(v[indices])
                if math.isnan(mean):
                    mean = 0
                if math.isnan(std):
                    std = 1

                kernel = np.ones(filter_size)
                counts = np.convolve(vals != 0, kernel, mode='same')

                s = np.convolve(vals, kernel, mode='same')
                s2 = np.convolve(vals ** 2, kernel, mode='same')
                local_var = (s2 - s ** 2 / counts) / (counts - 1)

                std2 = std ** 2
                np.nan_to_num(local_var, copy=False,
                              neginf=std2, posinf=std2, nan=std2)

                local_mean = s / counts
                local_mean[counts < 30] = mean
                local_var[counts < 30] = std2

                np.nan_to_num(local_mean, copy=False,
                              neginf=mean, posinf=mean, nan=mean)

                local_std = np.sqrt(local_var)
                vals[x[indices]] -= local_mean[x[indices]]
                vals[x[indices]] /= local_std[x[indices]]
                np.nan_to_num(vals, copy=False, nan=0, posinf=0, neginf=0)
                vals = vals * (1 + math.log(1 + mean, 30))
                pval_weights += [1 + math.log(1 + mean, 30)]
                v[indices] = vals[x[indices]]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            np.nan_to_num(v, copy=False, neginf=0, posinf=0, nan=0)
            distance_in_px = min(distance_in_px, n)
            for d in range(distance_in_px):
                indices = distances == d
                std = np.std(v[indices])
                mean = np.mean(v[indices])
                if math.isnan(mean):
                    mean = 0
                if math.isnan(std):
                    std = 1
                # print(std)
                v[indices] = (v[indices] - mean) / std
                np.nan_to_num(v, copy=False, nan=0, posinf=0, neginf=0)
    return pval_weights

#For parsing coordinate inputs
def parse_confirm(coord):
    coord = coord.strip().lower()

    if coord == "y":
        return True
    elif coord == "n":
        return False
    else:
        return parse_norm(input("Please only type y or n for your desired kernel:"))
        
#For parsing resolution input
def parse_res(coord):
    coord = coord.strip().lower()

    if coord.endswith("kb"):
        return int(coord[:-2]) * 1_000

    elif coord.endswith("mb"):
        return int(coord[:-2]) * 1_000_000

    else:
        raise ValueError("Format must be #mb or #kb")

#Ensures the correct units are entered into normalization type desired
def parse_norm(coord):
    coord = coord.strip().upper()

    if coord == "WEIGHT":
        return True

    elif coord == "NONE":
        return False
    else:
        return coord

def parse_kernel(coord):
    coord = coord.strip().lower()

    if coord == "dog":
        return True

    elif coord == "doh":
        return False
    else:
        return parse_norm(input("Please only type dog or doh for your desired kernel:"))

def parse_chrom(coord):
    if coord < 23:
        return str(coord)
    if coord == 23:
        return "X"
    if coord == 24:
        return "Y"
    if coord == 25:
        return "MT"

#Saves a set of loops for an individual chromosome in tab separated value format
def save_chrom(chrom_loops, chrom_it, res, pT, output_folder, mcool_name):
    cell_folder = output_folder / mcool_name
    cell_folder.mkdir(exist_ok=True)
    tsv_path = cell_folder / f"chr{chrom_it}_{res}_pt{pT}.tsv"

    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow([
            "chr",
            "bin1_start",
            "bin2_end",
            "chr",
            "bin2_start",
            "bin2_end",
            "sigma",
            "score",
            "p_value",
            "q_value"
        ])

        for chrom, bin1, bin2, sigma, score, pval, q in chrom_loops:
            writer.writerow([
                chrom,
                bin1 * res,
                (bin1 + 1) * res,
                chrom,
                bin2 * res,
                (bin2 + 1) * res,
                sigma,
                score,
                pval,
                q
            ])

#Let's begin

#Receiving user input to use to run the program
print("To begin, please place your desired .mcool file in your data folder:")
print("If you are making comparisons to MUSTACHE, please place your MUSTACHE output in the data folder in the following format:")
print("chr[chromosome number]_out[raw resolution divided by 1000]_pt[p-value threshold with no decimal].tsv")
print("\nPlease enter your desired input in the following format:")
print("<file_name.mcool> <integer> <integer> <#kb> <float> <normalization type> <dog/doh> <y/n> ")
print("\nHere is an example input:")
print("GSE63525_GM12878_diploid_maternal_copy_balance.mcool 5 5 5kb 0.1 kr doh y")
user_input = input(
    "\nEnter: <mcool_file> <start_chr> <end_chr> <resolution> <p_value> <norm> <dog/doh> <comparison to mustache>\n"
)

#Applying the input to variables
file_name, chr1, chr2, res, pT, norm_map, kern, compare = user_input.split()
chr1 = int(chr1)
chr2 = int(chr2)
res = parse_res(res)
pT = float(pT)
norm_map = parse_norm(norm_map)
kern = parse_kernel(kern)
compare = parse_confirm(compare)

#Path work - Input
project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "data"
data_folder.mkdir(exist_ok=True) #Even though it makes a folder if one does not exist, the program should have no data to run on if it does not exist there already.
base_images_folder = project_root / "images"
base_images_folder.mkdir(exist_ok=True)
images_folder = base_images_folder / file_name
images_folder.mkdir(exist_ok=True)
mcool_file = data_folder / file_name

#Path work - Output
#Ensures output from this program goes to the output folder in the project root
#If it is unavailable, make one automatically
out_dir = project_root / "output"
out_dir.mkdir(exist_ok=True)
if kern:
    file_path = out_dir / f"loops_{chr1}_{chr2}_{res}_{pT}_dog.csv"
else:
    file_path = out_dir / f"loops_{chr1}_{chr2}_{res}_{pT}_doh.csv"

#Loading the mcool files
file_load = f"{mcool_file}::resolutions/{res}"
hic_matrix = cooler.Cooler(file_load)

#Information about your data so far
print(f'Your cooler data resolution is {res}')
print("Chromosome names: \n")
print(list(hic_matrix.chromnames))
print("\n")

#Preparing all necessary arrays for output
all_loops = []
m_loops = []
all_match = []
no_match = []
all_apa = []

if compare:

    print("Reading MUSTACHE output files to compare later...")

    for i in range (chr1, chr2+1):
        chrom_it = parse_chrom(i)
        
        csv_file = data_folder / f"chr{chrom_it}_out{res // 1000}_pt{str(pT).replace(".", "")}.tsv"

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                chrom = str(row["BIN1_CHR"])
                bin1 = int(row["BIN1_START"]) // res
                bin2 = int(row["BIN2_START"]) // res

                m_loops.append((chrom, bin1, bin2))

else:
    del m_loops, all_match, no_match

for i in range(chr1, chr2+1):
    chrom_it = parse_chrom(i)

    gc.collect()

    print("Processing Chromosome ", chrom_it, "...")

    #Begin the timer for a chromosome loop
    program_start = time.perf_counter()

    #Obtain Balanced Matrix
    hic_numpy = hic_matrix.matrix(balance=norm_map).fetch(str(chrom_it))
    chrom_length = hic_matrix.chromsizes[str(chrom_it)]

    #Ensures no NaNs reach normalization
    hic_numpy = np.nan_to_num(
        hic_numpy,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Info on your chromosome's Hi-C Map
    print("Hi-C Matrix memory: ")
    print(hic_numpy.dtype)
    print(hic_numpy.nbytes / 1024**3, "GB")
    print("\nHi-C Matrix Class Type: ")
    print(type(hic_numpy))
    print("\nHi-C Matrix Shape: ")
    print(hic_numpy.shape)

    #Normalization

    #Organizing Matrix into Sparse Arguments and Preparing Boundaries
    x, y = np.nonzero(np.triu(hic_numpy))
    v = hic_numpy[x, y].astype(float)
    
    res = hic_matrix.binsize
    max_distpix = 2000000
    min_distpix = 50000
    max_loop_dist_bins = int(max_distpix / res)
    min_loop_dist_bins = int(min_distpix / res)

    if len(x) == 0 or len(y) == 0:
        print(f"No loops found for chromosome", chrom_it)
        continue

    print("\nNow normalizing matrix")
    p_weights = normalize_sparse(
        x,
        y,
        v,
        resolution=res,
        distance_in_px=int(max_distpix/res) 
    )
    print("Normalization Complete!\n")

    #Rebuilding the Matrix from sparse to a full map
    hic_numpy.fill(0)

    hic_numpy[x, y] = v
    hic_numpy[y, x] = v

    #Changing name from hic_numpy to normalized_hic
    normalized_hic = hic_numpy

    #Garbage collection
    del x
    del y
    del v
    del hic_numpy
    gc.collect()

    #Preparing for the subregion block processing
    shape = normalized_hic.shape
    r_start = 0
    r_end = 0
    c_start = 0
    c_end = 0
    all_blobs = []

    #Subregion Block size
    block_num = 0
    block_size = 2000
    overlap = 50
    step = block_size - overlap

    if kern:
        print ("Now searching for candidate blobs with Difference of Gaussian in full matrix using subregions\n")
    else:
        print ("Now searching for candidate blobs with Determinant of Hessian in full matrix using subregions\n")

    #Start of subregion processing

    loop_candidates = []

    for r_start in range(0, shape[0], step):
        r_end = min(r_start + block_size, shape[0])

        for c_start in range(0, shape[1], step):
            c_end = min(c_start + block_size, shape[1])

            block_min_dist = min(
                abs(r_start - c_end),
                abs(c_start - r_end),
                abs(r_start - c_start),
                abs(r_end - c_end)
            )

            if c_start > r_end + max_loop_dist_bins:
                continue

            if r_start > c_end + max_loop_dist_bins:
                continue

            subreg = normalized_hic[
                r_start:r_end,
                c_start:c_end
            ]

            start_bp = min(r_start * res, chrom_length)
            end_bp   = min(r_end   * res, chrom_length)

            #Performing Determinant of Hessian and Non-Maxima Suppresion
            if kern:
                response_volume = compute_dog_response(subreg)
            else:
                response_volume = compute_hessian_response(subreg)
            del subreg

            #Performing Non-Maxima Suppression
            keep = local_3d_max(response_volume)

            #Extracting candidates
            candidates = np.argwhere(keep)
        
            sigmas = np.geomspace(1, 8, 12)

            #Retrieving our candidates and ensuring they follow our rules and prevent
            #Duplicates across the diagonal
            for y, x, s in candidates:

                global_y = y + r_start
                global_x = x + c_start

                if global_x <= global_y:
                    continue

                dist = abs(global_y - global_x)

                if dist < min_loop_dist_bins:
                    continue

                if dist > max_loop_dist_bins:
                    continue

                score = response_volume[y, x, s]

                loop_candidates.append(
                    (
                        int(global_y),
                        int(global_x),
                        float(sigmas[s]),
                        float(score)
                    )
                )
            
            block_num += 1
            print(
                f"Block {block_num}: "
                f"rows {r_start}:{r_end}, "
                f"cols {c_start}:{c_end}, "
                f"blobs={len(candidates)}"
            )

    #End of subregion processing

    #No longer need the normalized matrix
    del normalized_hic
    gc.collect()

    #Instead, we use the raw Matrix
    raw_hic_numpy = hic_matrix.matrix(balance=False).fetch(str(chrom_it))

    #Remove NaNs
    raw_hic_numpy = np.nan_to_num(
        raw_hic_numpy,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    print("Subregion Processing Complete!")

    print(
        "Total candidates before p-values:",
        len(loop_candidates)
    )


    #This commented section is an optional choice for viewing data in distributions.
    #This was used to understand what is the best distribution for our
    #Determinant of Hessian and Difference of Gaussian scores

    # Group scores by sigma
    # scores_by_sigma = defaultdict(list)

    # for y, x, sigma, score in loop_candidates:
    #     scores_by_sigma[sigma].append(score)

    # for sigma in sorted(scores_by_sigma):

    #     scores = np.array(scores_by_sigma[sigma])

    #     plt.figure(figsize=(8,5))

    #     plt.hist(scores,
    #             bins=50,
    #             density=True,
    #             alpha=0.5,
    #             label="Observed")

    #     xvals = np.linspace(scores.min(), scores.max(), 500)

    #     # Laplace
    #     params = laplace.fit(scores)
    #     plt.plot(xvals,
    #             laplace.pdf(xvals, *params),
    #             label="Laplace")

    #     # Normal
    #     params = norm.fit(scores)
    #     plt.plot(xvals,
    #             norm.pdf(xvals, *params),
    #             label="Normal")

    #     # Exponential (only meaningful if scores are all positive)
    #     if np.all(scores >= 0):
    #         params = expon.fit(scores)
    #         plt.plot(xvals,
    #                 expon.pdf(xvals, *params),
    #                 label="Exponential")

    #     plt.title(f"Sigma = {sigma:.1f}")
    #     if kern:
    #         plt.xlabel("DoG Score")
    #     else:
    #         plt.xlabel("DoH Score")
    #     plt.ylabel("Density")
    #     plt.legend()

    #     plt.tight_layout()
    #     if kern:
    #         plt.savefig(os.path.join(images_folder, f"distribution_sigma_{sigma:.1f}_dog.png"), dpi=300)
    #     else:
    #         plt.savefig(os.path.join(images_folder, f"distribution_sigma_{sigma:.1f}_doh.png"), dpi=300)
    #     plt.close()

    # loop_candidates = sigma_nms(loop_candidates)

    #Obtaining p-values for the whole matrix
    if kern:
        print("Computing laplace parameters for DoG...")
        laplace_params = computing_laplace(loop_candidates)
        cand_pval = computing_pval_l(loop_candidates, laplace_params) #array with y,x,sigma,score,pval

    else:
        print("Computing exponential parameters for DoH...")
        expon_params = computing_exponential(loop_candidates)
        cand_pval = computing_pval_e(loop_candidates, expon_params) #array with y,x,sigma,score,pval

    #Considering loops next to each other directly and diagonally as the same
    #Using a Binary Matrix
    bin_rep = binary_matrix_global(cand_pval, shape)
    print(
        "After binary matrix:",
        len(bin_rep)
    )
    
    #Filtering out sparse candidate loops
    filtered_candidates = filter_sparse(shape[0], shape[1], bin_rep, raw_hic_numpy, st=0.8)
    print(
        "After sparsity filter:",
        len(filtered_candidates)
    )

    #Removing candidates that are less than 2 times their expected values
    diag_means = {}

    for d in range(raw_hic_numpy.shape[0]):

        diag = np.diagonal(
            raw_hic_numpy,
            offset=d
        )

        nonzero = diag[diag != 0]

        if len(nonzero) > 0:
            diag_means[d] = np.mean(nonzero)


    filtered = []

    for y, x, sigma, score, pval in filtered_candidates:

        d = x - y

        expected = diag_means[d]

        observed = raw_hic_numpy[y, x]

        if observed >= 2 * expected:
            filtered.append(
                (y, x, sigma, score, pval)
            )

    #These variables are no longer necessary
    del filtered_candidates
    del diag_means
    del cand_pval
    del bin_rep
    gc.collect()
    
    #Conducting Benjamini-Hochberg FDR with p-value scores
    pvals = [b[4] for b in filtered]
    reject, qvals, _, _ = multipletests(
        pvals,
        alpha=pT, #This is where the p-value threshold comes into play
        method="fdr_bh"
    )

    #Collecting the final set of loops
    final_loops = []

    for blob, keep, q in zip(
        filtered,
        reject,
        qvals
    ):
        if keep:
            final_loops.append((chrom_it,) + blob + (q,))
            all_loops.append((chrom_it,) + blob + (q,))

    #Time elapsed to run program
    elapsed = time.perf_counter() - program_start
    print(f"\nTotal runtime: {elapsed:.2f} seconds")
    print(f"Total runtime: {elapsed/60:.2f} minutes")

    chrom_apa = apa(raw_hic_numpy, final_loops)
    all_apa.append((chrom_it, chrom_apa))
    

    print("Final Loops: ", len(final_loops))

    del filtered
    del raw_hic_numpy
    gc.collect()

    #If you chose to compare to MUSTACHE output, your loops will be checked if they overlap with MUSTACHE
    if compare:

        match = []
        total_m = []

        for m in m_loops:
            if str(m[0]) == chrom_it:
                total_m.append(m)
        
        #b[1] = b's y
        #b[2] = b's x
        for b in final_loops:

            found = False

            #m[0] = chromosome number
            #m[1] = m's y
            #m[2] = m's x
                
            for m in m_loops:
                if str(m[0]) != chrom_it:
                    continue

                

                if abs(m[1] - b[1]) <= 5 and abs(m[2] - b[2]) <= 5:
                    match.append(b)
                    all_match.append(b)
                    found = True
                    break

            if not found:
                no_match.append((b))

        print("Number of matched loops:", len(match))
        print("Total number of MUSTACHE loops: ", len(total_m))

    del reject
    del qvals

    gc.collect()

    save_chrom(final_loops, chrom_it, res, pT, out_dir, file_name)


#Output
print(f"Number of loops in Chr{parse_chrom(chr1)}-{parse_chrom(chr2)}: {len(all_loops)}")

if compare:
    print(f"Number of loops matched with MUSTACHE in Chr{parse_chrom(chr1)}-{parse_chrom(chr2)}: {len(all_match)}")
    print("Number of loops not overlapping: ", len(no_match))

with open(file_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Category",
        "Chromosome",
        "bin1",
        "bin2",
        "sigma",
        "DoH score",
        "p_value",
        "q"
    ])

    for row in all_loops:
        writer.writerow(["all", *row])
    if compare:
        for row in all_match:
            writer.writerow(["matched", *row])
        for row in no_match:
            writer.writerow(["unmatched", *row])

    # Blank line
    writer.writerow([])

    # Summary marker
    writer.writerow(["SUMMARY"])

    writer.writerow(["All Loops", len(all_loops)])
    if compare:
        writer.writerow(["Matched Loops", len(all_match)])
        writer.writerow(["Unmatched Loops", len(no_match)])
        writer.writerow(["Total MUSTACHE Loops", len(m_loops)])
    writer.writerow(["APA Score", chrom_apa])


#For Plotting
#Shows every chromosome's best window of loops found
print("Saving images of the most dense windows of loops in each chromosome")
for i in range(chr1,chr2+1):
    chrom_it = parse_chrom(i)

    chrom_loops = []

    for l in all_loops:
        if l[0] == chrom_it:
            chrom_loops.append((l[1], l[2]))   # absolute bin numbers

    if not chrom_loops:
        continue

    chrom_sizes = dict(zip(
        hic_matrix.chromnames,
        hic_matrix.chromsizes
    ))

    chrom_size = chrom_sizes[str(chrom_it)]
    max_bin = chrom_size // res

    window = min(200, max_bin)
    half = window // 2

    best_count = -1
    best_y = 0
    best_x = 0

    for center_y, center_x in chrom_loops:

        count = 0

        for y, x in chrom_loops:
            if (
                center_y - half <= y < center_y + half and
                center_x - half <= x < center_x + half
            ):
                count += 1

        if count > best_count:
            best_count = count

            best_y = center_y - half
            best_x = center_x - half

    # Keep the window inside the chromosome
    best_y = max(0, min(best_y, max_bin - window))
    best_x = max(0, min(best_x, max_bin - window))

    start_y = best_y * res
    end_y   = min((best_y + window) * res, chrom_size)

    start_x = best_x * res
    end_x   = min((best_x + window) * res, chrom_size)

    # Fetch exactly that square
    hic = hic_matrix.matrix(balance=False).fetch(
        f"{chrom_it}:{start_y}-{end_y}",
        f"{chrom_it}:{start_x}-{end_x}"
    )

    region_loops = []

    for y, x in chrom_loops:
        if (
            best_y <= y < best_y + window and
            best_x <= x < best_x + window
        ):
            region_loops.append((y, x))

    if compare:
        must_loops = []

        for c, y, x in m_loops:
            if c != chrom_it:
                continue

            if (
                best_y <= y < best_y + window and
                best_x <= x < best_x + window
            ):
                must_loops.append((y, x))
    
    #MatPlot

    plt.figure(figsize=(8,8))
    extent = [start_x, end_x, end_y, start_y]

    im = plt.imshow(hic, origin="upper", cmap="Reds", extent=extent)    

    ticks_x = np.linspace(start_x, end_x, 5)
    ticks_y = np.linspace(start_y, end_y, 5)

    plt.xticks(
        ticks_x,
        [f"{t/1e6:.2f}M" for t in ticks_x]
    )

    plt.yticks(
        ticks_y,
        [f"{t/1e6:.2f}M" for t in ticks_y]
    )


    # Plot your detected loops
    for y, x in region_loops:
        plt.scatter(
            x * res,
            y * res,
            s=15,
            facecolors="none",
            edgecolors="blue",
            linewidths=0.8
        )

    # Plot MUSTACHE loops
    if compare:
        for y, x in must_loops:
            plt.scatter(
                x * res,
                y * res,
                marker="x",
                c="green",
                s=15,
                linewidths=0.8
            )

        # Legend keys
        plt.scatter(
            [],
            [],
            s=15,
            facecolors="none",
            edgecolors="blue",
            label="HiCinBoRGH loops"
        )

        plt.scatter(
            [],
            [],
            marker="x",
            c="green",
            s=15,
            linewidths=0.8,
            label="MUSTACHE loops"
        )

        plt.legend(loc="upper right", fontsize=10)

    plt.xlabel(f"Chr{chrom_it} Position (Mb)")
    plt.ylabel(f"Chr{chrom_it} Position (Mb)")
    plt.colorbar(im)
    plt.tight_layout()
    if kern:
        plt.savefig(os.path.join(images_folder, f"chr{chrom_it}_out{res}_pT{pT}_dog.png"), dpi=300)
    else:
        plt.savefig(os.path.join(images_folder, f"chr{chrom_it}_out{res}_pT{pT}_doh.png"), dpi=300)
    plt.close()

    print(f"Saving Chromosome {chrom_it}...")
    print(f"Window in bins:")
    print(f"Rows: {best_y} - {best_y+window}")
    print(f"Cols: {best_x} - {best_x+window}")
    print(f"Loops in window: {best_count}")



