# VintedHomework
Homework assignment task for Vinted Machine Learning Engineer position


## Steps to execute
 1. Build a docker image with `docker build https://github.com/DonatasTamosauskas/VintedHomework.git`
 2. Create a container from the image and run it exposing port 5000

## The api has two paths:
 - /category
 - /tag


Both of the paths accept POST request with the image file attached (attribute name `image`)  
