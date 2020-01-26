from shotTypeML_pkg import ShotTypeClassifierML

# Create the classifier
stc = ShotTypeClassifierML.ShotTypeClassifierML('demo_config.yaml')

# Classify a single image
csvResult = stc.predict('data/images/image0.png', False, 'results/result_single_image.csv')
print(csvResult)

# Classify all images within a folder
csvResult = stc.predict('data/images/', False, 'results/result_multiple_images.csv')
print(csvResult)

# Classify a single video
csvResult = stc.predict('data/videos/video0.mp4', True, 'results/result_single_video.csv')
print(csvResult)

# Classify all videos within a folder
csvResult = stc.predict('data/videos/', True, 'results/result_multiple_videos.csv')
print(csvResult)
