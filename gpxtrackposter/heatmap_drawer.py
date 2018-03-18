# Copyright 2016-2018 Florian Pigorsch & Contributors. All rights reserved.
#
# Use of this source code is governed by a MIT-style
# license that can be found in the LICENSE file.

import argparse
import logging
import math
import svgwrite
import s2sphere as s2
from .exceptions import ParameterError
from .poster import Poster
from .tracks_drawer import TracksDrawer
from .xy import XY
from . import utils

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



log = logging.getLogger(__name__)


class HeatmapDrawer(TracksDrawer):
    def __init__(self, the_poster: Poster):
        super().__init__(the_poster)
        self._center = None
        self._radius = None
        self._cluster = True

    def create_args(self, args_parser: argparse.ArgumentParser):
        group = args_parser.add_argument_group('Heatmap Type Options')
        group.add_argument('--heatmap-center', dest='heatmap_center', metavar='LAT,LNG', type=str,
                           help='Center of the heatmap (default: automatic).')
        group.add_argument('--heatmap-radius', dest='heatmap_radius', metavar='RADIUS_KM', type=float,
                           help='Scale the heatmap such that at least a circle with radius=RADIUS_KM is visible '
                                '(default: automatic).')

    def fetch_args(self, args: argparse.Namespace):
        self._center = None
        if args.heatmap_center:
            latlng_str = args.heatmap_center.split(',')
            if len(latlng_str) != 2:
                raise ParameterError('Not a valid LAT,LNG pair: {}'.format(args.heatmap_center))
            try:
                lat = float(latlng_str[0].strip())
                lng = float(latlng_str[1].strip())
            except ValueError as e:
                raise ParameterError('Not a valid LAT,LNG pair: {}'.format(args.heatmap_center)) from e
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                raise ParameterError('Not a valid LAT,LNG pair: {}'.format(args.heatmap_center))
            self._center = s2.LatLng.from_degrees(lat, lng)
        if args.heatmap_radius:
            if args.heatmap_radius <= 0:
                raise ParameterError('Not a valid radius: {} (must be > 0)'.format(args.heatmap_radius))
            if not args.heatmap_center:
                raise ParameterError('--heatmap-radius needs --heatmap-center')
            self._radius = args.heatmap_radius

    def _determine_bbox(self) -> s2.LatLngRect:
        if self._center:
            log.info('Forcing heatmap center to {}'.format(self._center))
            dlat, dlng = 0, 0
            if self._radius:
                er = 6378.1
                quarter = er * math.pi / 2
                dlat = 90 * self._radius / quarter
                scale = 1 / math.cos(self._center.lat().radians)
                dlng = scale * 90 * self._radius / quarter
            else:
                for tr in self.poster.tracks:
                    for line in tr.polylines:
                        for latlng in line:
                            d = abs(self._center.lat().degrees - latlng.lat().degrees)
                            dlat = max(dlat, d)
                            d = abs(self._center.lng().degrees - latlng.lng().degrees)
                            while d > 360:
                                d -= 360
                            if d > 180:
                                d = 360 - d
                            dlng = max(dlng, d)
            return s2.LatLngRect.from_center_size(self._center, s2.LatLng.from_degrees(2 * dlat, 2 * dlng))

        elif self._cluster:
            for tr in self.poster.tracks:
                log.info(tr.bbox().get_center())
            coords = [(tr.bbox().get_center().lat().degrees, tr.bbox().get_center().lng().degrees) for tr in self.poster.tracks]
            X = np.array(coords)
            #print(X)
            db = DBSCAN(eps=0.3, min_samples=10).fit(X)
            #print(X.size)
            cluster_results = np.empty((db.labels_.size, 3))
            #print(len(cluster_results), len(db.labels_))
            for i,entry in enumerate(cluster_results):
                #print(cluster_results[i], db.labels_[i])
                cluster_results[i] = np.append(X[i], db.labels_[i])
            #print(cluster_results)
            tracks_bbox = s2.LatLngRect()
            for i,tr in enumerate(self.poster.tracks):
                print(int(cluster_results[i][2]))
                if int(cluster_results[i][2]) == 0:
                    tracks_bbox = tracks_bbox.union(tr.bbox())
            return tracks_bbox



            # print(db.labels_)
        
            # print(db.components_)
            # print(db.labels_)
            # print(len(db.labels_), len(db.components_), len(db.labels_))
            # print(type(db.components_[0]))

            # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            # core_samples_mask[db.core_sample_indices_] = True
            # labels = db.labels_
            # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            # print('Estimated number of clusters: %d' % n_clusters_)



            # # Black removed and is used for noise instead.
            # unique_labels = set(labels)
            # colors = [plt.cm.Spectral(each)
            #           for each in np.linspace(0, 1, len(unique_labels))]
            # for k, col in zip(unique_labels, colors):
            #     if k == -1:
            #         # Black used for noise.
            #         col = [0, 0, 0, 1]

            #     class_member_mask = (labels == k)

            #     xy = X[class_member_mask & core_samples_mask]
            #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            #              markeredgecolor='k', markersize=14)

            #     xy = X[class_member_mask & ~core_samples_mask]
            #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            #              markeredgecolor='k', markersize=6)

            # plt.title('Estimated number of clusters: %d' % n_clusters_)
            # plt.show()


        tracks_bbox = s2.LatLngRect()
        for tr in self.poster.tracks:
            tracks_bbox = tracks_bbox.union(tr.bbox())
        return tracks_bbox

    def draw(self, dr: svgwrite.Drawing, size: XY, offset: XY):
        normal_lines = []
        special_lines = []
        bbox = self._determine_bbox()
        for tr in self.poster.tracks:
            for line in utils.project(bbox, size, offset, tr.polylines):
                if tr.special:
                    special_lines.append(line)
                else:
                    normal_lines.append(line)
        for lines, color in [(normal_lines, self.poster.colors['track']),
                             (special_lines, self.poster.colors['special'])]:
            for opacity, width in [(0.1, 5.0), (0.2, 2.0), (1.0, 0.3)]:
                for line in lines:
                    dr.add(dr.polyline(points=line, stroke=color, stroke_opacity=opacity, fill='none',
                                       stroke_width=width, stroke_linejoin='round', stroke_linecap='round'))


