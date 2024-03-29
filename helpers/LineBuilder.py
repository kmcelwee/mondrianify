from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import mode
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean as distance

class LineBuilder:
    """Create the segments from an image"""
    def __init__(self, image_in, min_percent_split=.1):
        self.image_in = image_in
        self.min_percent_split = min_percent_split

        im = Image.open(image_in)
        self.width = im.width
        self.height = im.height

        n = np.array(im)
        self.pos_ids = np.argwhere(n == 250)
        self.all_x = self.pos_ids[:, 1]
        self.all_y = self.pos_ids[:, 0]

        # Vars to set later
        self.segments = None
        self.raw_segments = None
        self.kmeansy = None
        self.kmeansx = None


    def get_best_kmeans(self, k_range=(2, 7)):
        """Run kmeans models on the x axis and the y axis for the given k_range"""
        def get_top_models(kmeans_models, n=5):
            """Given a list of kmeans models, use "max percent difference" as a 
            heuristic to determine which model has the appropriate number of clusters.
            """
            def perc(a, b):
                """get the percent difference between two values"""
                return abs(a - b)/((a+b)*.5)
            
            inertias = [x.inertia_ for x in kmeans_models]
            perc_diff = [perc(inertias[x], inertias[x+1]) for x, _ in enumerate(inertias[:-1])]
            diff_perc_diff = [perc_diff[x] - perc_diff[x+1] for x, _ in enumerate(perc_diff[:-1])]
            
            max_diff_perc_diff = sorted(diff_perc_diff, key=lambda x: -x)[:n]
            max_diff_indices = [diff_perc_diff.index(x) for x in max_diff_perc_diff]
            
            model_indices = [x+1 for x in max_diff_indices]
            
            return [m for i, m in enumerate(kmeans_models) if i in model_indices]

        # Create a kmeans model for x and y with n_clusters equal to each value in k_range
        all_kmeansy = [KMeans(n_clusters=i).fit(self.all_y.reshape(-1, 1)) for i in range(*k_range)]
        all_kmeansx = [KMeans(n_clusters=i).fit(self.all_x.reshape(-1, 1)) for i in range(*k_range)]

        top_kmeansy = get_top_models(all_kmeansy)
        top_kmeansx = get_top_models(all_kmeansx)

        self.kmeansy = top_kmeansy[0]
        self.kmeansx = top_kmeansx[0]


    def get_raw_segments(self, buffer_quantile=0.05):
        """Combine logic with the kmeans models to get the raw segments"""
        def threshold_split(input_l, min_percent_split):
            """Split an input list if the difference between any values is 
            greater than the given threshold
            """
            max_diff = int(min_percent_split * self.width)
            return_l = []

            # Make sure the list is sorted
            input_l = sorted(input_l)
            
            # Add values to `contiguous` unless the difference between the values
            #   is too great, then append to the return list.
            contiguous = []
            for i, val in enumerate(input_l[:-1]):
                if input_l[i+1] - input_l[i] > max_diff:
                    return_l.append(contiguous)
                    contiguous = []
                else:
                    contiguous.append(val)
            if contiguous != []:
                return_l.append(contiguous)
            
            # Filter out any lists that are only one point long
            return_l = [x for x in return_l if len(x) > 1]

            return return_l

        raw_segments = []
        all_x, all_y = self.all_x, self.all_y

        kmeans = self.kmeansx
        for cc_x, label in zip(kmeans.cluster_centers_, range(kmeans.n_clusters)):
            # get the mode of the x values in this cluster
            group_x = all_x[kmeans.labels_ == label]
            m = mode(group_x).mode[0]
            group_x = group_x.tolist()

            # get all the y values that correspond to the x values in this cluster
            y_bound = [p[0] for p in self.pos_ids if p[1] in group_x]
            assert len(y_bound) == len(group_x)

            # Use the range of those y values to create the segments
            #   Use `buffer_quantile` as a way to drop outliers and `threshold_split`
            #   as a way to help split up multimodal distributions
            y_bounds = threshold_split(y_bound, self.min_percent_split)
            for yb in y_bounds:
                raw_segments.append([[m, np.quantile(yb, buffer_quantile)], [m, np.quantile(yb, 1-buffer_quantile)]])

        # repeat for y
        kmeans = self.kmeansy
        for cc_y, label in zip(kmeans.cluster_centers_, range(kmeans.n_clusters)):
            group_y = all_y[kmeans.labels_ == label]
            m = mode(group_y).mode[0]
            group_y = group_y.tolist()
            x_bound = [p[1] for p in self.pos_ids if p[0] in group_y]
            assert len(x_bound) == len(group_y)
            x_bounds = threshold_split(x_bound, self.min_percent_split)
            for xb in x_bounds:
                raw_segments.append([[np.quantile(xb, buffer_quantile), m], [np.quantile(xb, 1-buffer_quantile), m]])

        self.raw_segments = raw_segments


    def clean_raw_segments(self):
        def intersects_y(y, seg):
            low_y, high_y = min(seg[1][1], seg[0][1]), max(seg[1][1], seg[0][1])
            return low_y <= y <= high_y

        def intersects_x(x, seg):
            low_x, high_x = min(seg[1][0], seg[0][0]), max(seg[1][0], seg[0][0])
            return low_x <= x <= high_x
        
        width = self.width
        height = self.height

        # new_segments will be the final framework for our painting's lines
        new_segments = {
            'x': [[[0, 0], [0, height]], [[width, 0], [width, height]]],
            'y': [[[0, 0], [width, 0]], [[0, height], [width, height]]]
        } 

        # a shorthand for "any index"
        a = 0

        # sort raw segments in descending order by size so that we prioritize larger segments
        raw_segments_sorted = sorted(self.raw_segments, key=lambda x: -distance(x[0], x[1]))

        # extend each raw segment to the closest existing line
        for seg in raw_segments_sorted:
            (x1, y1), (x2, y2) = seg
            if y1 == y2:
                y = y1
                x1, x2 = min([x1, x2]), max([x1, x2])
                
                # get the closest vertical segment that
                new_x1 = min([
                    { 'new_x': seg[a][0], 'delta': abs(x1 - seg[a][0]) }
                 for seg in new_segments['x'] if intersects_y(y, seg)
                ], key=lambda t: t['delta'])['new_x']
                
                # repeat the process, but don't let the segment collapse to a point.
                new_x2 = min([
                    { 'new_x': seg[a][0], 'delta': abs(seg[a][0] - x2) }
                 for seg in new_segments['x'] if seg[a][0] != new_x1 and intersects_y(y, seg)
                ], key=lambda t: t['delta'])['new_x']
                
                new_segments['y'].append([[new_x1, y], [new_x2, y]])
            
            # repeat for vertical segments
            else:
                x = x1
                y1, y2 = min([y1, y2]), max([y1, y2])
                
                new_y1 = min([{ 'new_y': seg[a][1], 'delta': abs(y1 - seg[a][1]) }
                 for seg in new_segments['y'] if intersects_x(x, seg)
                ], key=lambda t: t['delta'])['new_y']
                
                new_y2 = min([{ 'new_y': seg[a][1], 'delta': abs(seg[a][1] - y2) }
                 for seg in new_segments['y'] if seg[a][1] != new_y1 and intersects_x(x, seg)
                ], key=lambda t: t['delta'])['new_y']
                
                new_segments['x'].append([[x, new_y1], [x, new_y2]])
        
        self.segments = new_segments


    def analyze_image(self):
        """Usher image through pipeline"""    
        self.get_best_kmeans()
        self.get_raw_segments()
        self.clean_raw_segments()


    def create_histogram(self, filename, hist_size=0.65, fig_size=8):
        """Put histograms on the axes and sketch the raw segments to help 
        understand what the kmeans models are doing.
        """
        def draw_raw_segments(raw_segments, ax):
            """Place the raw segments onto the given axis"""
            for seg in raw_segments:
                (x1, y1), (x2, y2) = seg
                width = self.width
                height = self.height

                # matplotlib draws lines as percentages and the y axis is flipped
                #   so we need to divide by height and width, and subtract the value
                #   from 1 for vertical segments
                if x1 == x2:
                    ax.axvline(linestyle='--', x=x1, color='gray', ymin=(1-y1/height), ymax=(1-y2/height))
                else:
                    ax.axhline(linestyle='--', y=y1, color='gray', xmin=(x1/width), xmax=(x2/width))

        x = self.all_x
        y = self.all_y

        if self.width > self.height:
            hist_width = hist_size
            hist_height = hist_width / self.width * self.height
        else:
            hist_height = hist_size
            hist_width = hist_height / self.height * self.width
        left = 0.1
        bottom = 0.1

        rect_scatter = [left, bottom, hist_width, hist_height]
        rect_histx = [left, bottom + hist_height, hist_width, 0.2]
        rect_histy = [left + hist_width, bottom, 0.2, hist_height]

        fig = plt.figure(figsize=(fig_size, fig_size))

        ax = fig.add_axes(rect_scatter)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False) 

        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)

        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['right'].set_visible(False)
        ax_histx.spines['left'].set_visible(False)
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['bottom'].set_visible(False)

        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histx.tick_params(axis="y", labelleft=False, left=False)
        ax_histy.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        ax.imshow(Image.open(self.image_in), cmap='Greys')

        bin_proportion = 0.6
        binsx = int(self.width*bin_proportion)
        binsy = int(self.height*bin_proportion)
        
        ax_histx.hist(x, bins=binsx, color="k")
        ax_histy.hist(y, bins=binsy, orientation='horizontal', color="k")
        
        draw_raw_segments(self.raw_segments, ax)

        fig.savefig(filename)
        plt.close('all')

    def save(self, filename):
        """Save the histogram to filename"""
        self.create_histogram(filename)
