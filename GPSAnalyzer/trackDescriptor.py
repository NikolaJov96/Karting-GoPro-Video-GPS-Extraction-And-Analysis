import sys

class TrackDescriptor:
    """
    Class describing track start and sector edge lines.
    """

    def __init__(self, input_file_path) -> None:
        """
        Creates the track descriptor object by loading a file
        """
        with open(input_file_path, 'r') as input_file:
            self.track_name = input_file.readline().strip()
            self.start_line = [float(x) for x in input_file.readline().strip().split(',')]

            self.sector_lines = []
            input_line = input_file.readline()
            while input_line != '':
                self.sector_lines.append([float(x) for x in input_line.strip().split(',')])
                input_line = input_file.readline()


if __name__ == '__main__':
    track_descriptor = TrackDescriptor(sys.argv[1])
    print(track_descriptor.track_name)
    print(track_descriptor.start_line)
    print(track_descriptor.sector_lines)
