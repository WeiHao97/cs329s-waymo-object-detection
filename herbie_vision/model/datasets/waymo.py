from PIL import Image
import numpy as np
from herbie_vision.utils.image import annotations_to_df, process_resizing
from herbie_vision.utils.gcp_utils import download_blob ,upload_blob

    

CATEGORY_NAMES = ['TYPE_VEHICLE','TYPE_PEDESTRIAN','TYPE_CYCLIST']
CATEGORY_IDS = [1,2,4]

class WaymoDataset(data.Dataset):
    def __init__(self, gcp_bucket, gcp_annotations_path, local_path_to_images, 
                 local_path_to_processed_images, cat_names, cat_ids):
        super(WaymoDataset, self).__init__()
        
        # filepaths
        self.gcp_bucket = gcp_bucket
        self.gcp_annotations_path = gcp_annotations_path
        self.local_path_to_images = local_path_to_images
        self.local_path_to_processed_images = local_path_to_processed_images
        
        # high level summary values
        self.num_classes = len(cat_names)
        self.category_names = cat_names
        self.category_ids = cat_ids
        
        
        # setup data directory
        if os.path.exists('./data')==False:
            os.mkdir('./data')
            os.mkdir(self.path_to_images)
            os.mkdir(self.path_to_processed_images)
        
        
        # read in annotations
        client = storage.Client()
        bucket = client.get_bucket(self.gcp_bucket)
        
        download_blob(self.gcp_bucket,
                           self.gcp_annotations_path,
                           './data/annotations.json')
        
        f = open('./data/annotations.json','r')
        self.annotations = json.load(f)
        f.close()
        
        # convert annotations to dataframe
        annotations_df = pd.DataFrame(annotations['annotations'])
        annotations_df['x'] = annotations_df['bbox'].apply(lambda x: x[0])
        annotations_df['y'] = annotations_df['bbox'].apply(lambda x: x[1])
        annotations_df['width'] = annotations_df['bbox'].apply(lambda x: x[2])
        annotations_df['height'] = annotations_df['bbox'].apply(lambda x: x[3])
        annotations_df.drop(columns='bbox')
        self.annotations_df = annotations_df

        
        # determine segment paths
        self.segment_paths = []
        for image in self.annotations['images']:
            uri = image['gcp_url']
            segment = '/'.join(uri.split('/')[3:7])+'/'
            if segment not in self.segment_paths:
                self.segment_paths.append(segment)
        
        
        # Download images for segments to local folder
        for segment in self.segment_paths:
            blobs = bucket.list_blobs(prefix=segment, delimiter='/')
            for blob in list(blobs):
                filename=blob.name.replace(segment,'')
                blob.download_to_filename('./data/images/{}'.format(filename))


        # Preprocess images to be the same size
        self.annotations_df = process_resizing(self.path_to_processed_images, annotations_df,800)
        

    def __getitem__(self, idx):
        image_url = self.annotations['images'][idx]['gcp_url']
        filename = image_url.split('/')[-1]
        image = Image.open(self.path_to_processed_images+'{}'.format(filename))
        image = np.asarray(img, dtype="float64") / 255.
        image = torch.tensor(image).permute(2,0,1)        
        
        return image
        
    
    def __len__(self):
        return len(annotations['images'])