import datetime
import json
import os
import sys

from gpx_converter import Converter


class FormatConverter:
    """
    Class responsible for converting other formats to GEOJSON file
    """

    @staticmethod
    def date_to_utc(date):
        """
        Convert datetime object to utc time
        """

        return int(datetime.datetime.timestamp(date) * 1000)

    @staticmethod
    def convert_gpx_to_geojson(gpx_file, geojson_file):
        """
        Convert GPX file to GEOJSON file
        """

        gpx = Converter(input_file=gpx_file).gpx_to_dictionary()

        points_num = len(gpx['time'])

        assert \
            points_num == len(gpx['time']) and \
            points_num == len(gpx['latitude']) and \
            points_num == len(gpx['longitude']) and \
            points_num ==  len(gpx['altitude'])

        geojson = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': []
            },
            "properties": {
                "device": "HERO8 Black",
                "AbsoluteUtcMicroSec": [],
                "RelativeMicroSec": []
            }
        }

        for i in range(points_num):

            # Add location
            geojson['geometry']['coordinates'].append([
                gpx['longitude'][i],
                gpx['latitude'][i],
                gpx['altitude'][i]
            ])

            # Add absolute UTC time
            geojson['properties']['AbsoluteUtcMicroSec'].append(
                FormatConverter.date_to_utc(gpx['time'][i])
            )

            # Add relative milliseconds time
            geojson['properties']['RelativeMicroSec'].append(
                FormatConverter.date_to_utc(gpx['time'][i]) -
                FormatConverter.date_to_utc(gpx['time'][0])
            )

        with open(geojson_file, 'w') as out_file:
            json.dump(geojson, out_file)


if __name__ == '__main__':
    """
    Convert GPX file to GEOJSON
    """

    # Check command line arguments
    if len(sys.argv) != 3:
        print('Usage: %s <gpx_file> <out_geojson_file>' % os.path.basename(__file__))
        exit(1)

    FormatConverter.convert_gpx_to_geojson(sys.argv[1], sys.argv[2])
