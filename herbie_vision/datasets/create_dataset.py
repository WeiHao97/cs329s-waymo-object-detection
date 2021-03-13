import pandas as pd
import os
from herbie_vision.utils.gcp_utils import download_blob
from herbie_vision.utils.train_utils import concatenateJSON

def segmentByDate(df, startdate, enddate):
    """
    segments by date

    :param: df: pandas dataframe
    :param startdate: (inclusive) start date for paths -- string data type
    :param enddate: (exclusive) start date for paths -- string data type
    :return: dataframe containing segments in provided date range

    *** sample usage: segment = segmentByDate(df, '2019-02-10','2019-03-13')
    """

    #converting string objects in 'date' column to datetime objects
    pd.to_datetime(df['date'])

    segment = df[(df['date'] >= startdate) & (df['date'] <= enddate)]
    return segment

def segmentTimeOfDay(df, timeOfDay):
    """
    segments by time of day

    :param: df: pandas dataframe
    :param timeOfDay: time of day string (day, night, etc.)
    :return: dataframe containing segments that have specified time of day
    """

    return df[(df['time_of_day'] == timeOfDay)]

def segmentLocation(df, location):
    """
    segments by location

    :param: df: pandas dataframe
    :param location: location string (location_sf)
    :return: dataframe containing segments that have specified location
    """

    return df[(df['location'] == location)]

def equalDist(df, columns=['time_of_day', 'location'], upsample=False, num_samples_per_type=-1):
    """

    :param: df: pandas dataframe
    :param columns: list of columns where equal distribution of samples is desired
    :param upsample: boolean value deciding whether upsampling is desired **CURRENTLY NOT IMPLEMENTED**
    :param num_samples_per_type: number of samples per type, must be given if upsample = True
    :return: dataframe containing segments with equal distribution of samples
    """
    if upsample == True and num_samples_per_type == -1:
        raise Exception("If upsample is false, must provide number of samples per type")

    #num samples per type will automatically be set to minimum quantity type size if upsample = False

    #possible bug if the random sample cuts out a sample of other columns but i was unsure of how to work around this
    for col in columns:

        #**if we want upsample of segments which i did not think we did? not super sure but can finish the upsample part if needed
        segment = []

        smallest_sample_size = df[col].value_counts().min()
        if upsample == False:
            num_samples_per_type = smallest_sample_size
        if num_samples_per_type > smallest_sample_size:
            raise Exception('Number of samples per type cannot exceed smallest sample size ')
        for val in df[col].unique():
            segment.append(df[(df[col] == val)].sample(n=num_samples_per_type, random_state=1))

        df = pd.concat(segment)

    return df


def dataframeToPaths(df):
    """
    returns list of paths for given dataframe

    :param df: dataframe/segment of dataframe
    :return: gcp urls/paths for this segment/dataframe
    """
    return df['gcp_url'].tolist()



if __name__ == "__main__":
    #setup, will move to reading cloud csv blob later
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/tiffanyshi/Desktop/waymo-2d-object-detection-9ea7bd3b9e0b.json'
    root_dir = '/Users/tiffanyshi/PycharmProjects/329swaymoproject/herbie-vision/'

    download_blob('waymo-processed',
                          'train/metadata/metadata.csv',
                          root_dir + "herbie_vision/datasets" + '/' + 'metadata.csv')
    df = pd.read_csv('metadata.csv')

    #move to command line later --> this is the editable lines where we can choose our data segments
    df = segmentByDate(df, '2017-02-10','2019-03-13')
    df = segmentLocation(df, 'location_phx')
    df = equalDist(df, columns=['time_of_day'])

    #necessary to input into concatenateJSON()
    df['gcp_url'] = df['gcp_url'].str.replace('gs://waymo-processed/', '')
    tests = dataframeToPaths(df)
    concatenateJSON(tests, root_dir, "test", "tester.json") #the uploading is very slow but i don't think i can control that
    os.remove('metadata.csv')

