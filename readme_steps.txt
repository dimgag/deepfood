# ACTIVATE THE ENVIRONMENT
ipython kernel install --user --name=denv


# DOWNLOAD THE DATA
!wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz


# UNZIP .tar.gz file

!tar xzvf food-101.tar.gz


# Move file to another dir
mv ~/Downloads/MyFile.txt ~/Documents/Work/MyFile.txt