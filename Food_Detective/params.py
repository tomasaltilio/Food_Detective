# File with variables stored
import pandas as pd
import os

# Prediction categories
path = os.path.join(os.getcwd(), 'model-export-tflite/dict.txt')
cats = pd.read_csv(path, header=None)
categories = cats.values.tolist()

# CSS to include background images
background_image = """
    h1 {
    color: black;
    }
    body {
    background-image: url(https://scontent.faep8-1.fna.fbcdn.net/v/t1.0-9/118764930_137029078090158_291676113316331324_o.jpg?_nc_cat=105&ccb=1-3&_nc_sid=e3f864&_nc_ohc=LRcKIE6_Hs8AX9XzwIO&_nc_ht=scontent.faep8-1.fna&oh=289cff6da615877c55a279de7ddffcd6&oe=60769602);
    background-size: cover;
    }
    """
# Radio button default to be empty 
radio_button = """ div[role="radiogroup"] >  :first-child{
    display: none !important;
    }
    """
    
# Removing background image for mobile devices
responsive = '''
@media (max-width: 960px) { 
        body {
            background-image: none !important; 
      }
}
'''