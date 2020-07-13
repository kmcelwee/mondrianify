# mondrianify
A pipeline for turning images into paintings by Piet Mondrian. Interested in seeing examples? Check out the Twitter bot [@PietMondrianAI](https://twitter.com/PietMondrianAI) and its respective the repo [mondrian-twitter]](https://github.com/kmcelwee/mondrian-twitter/)].

![Mondrianify flowchart](flowchart.png)

### Getting setup locally
Python version 3.7, run `pip install -r requirements.txt`. Then running `python MondrianPipeline.py` will draw a random photo from [Unsplash](https://unsplash.com/developers) and apply the transformation. The default directory `output` will be created and the image files will be placed inside.

### MondrianPipline.py
The overarching class to help usher an image through the entire transformation. As it steps through the pipeline, it periodically saves the images output by the helper classes to a defined output directory. It relies on the classes in `helpers` to complete most phases of the process.

### Helpers
#### BorderBuilder.py
Helps apply Holisticly-Nested Edge Detection to an image so that we can get the major features of an image.

#### ColorBuilder.py
Determines the colors used in a Mondrian painting. It draws from `colors.py`, a file created by sampling from Mondrian's palette.

#### LineBuilder.py
Create many [KMeans models](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) to get a rough sketch of the segments that define an image. Then build out a Mondrian framework from those sketches.

#### Painting.py
Combines the LineBuilder and ColorBuilder classes to create the final Mondrian painting.