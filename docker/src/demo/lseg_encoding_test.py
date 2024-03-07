import torch
import encoding

# Get the model
model = encoding.models.get_model('Encnet_ResNet50s_PContext', pretrained=True).cuda()
model.eval()

# Prepare the image
url = 'https://github.com/zhanghang1989/image-data/blob/master/' + \
      'encoding/segmentation/pcontext/2010_001829_org.jpg?raw=true'
filename = 'example.jpg'
img = encoding.utils.load_image(
    encoding.utils.download(url, filename)).cuda().unsqueeze(0)

# Make prediction
output = model.evaluate(img)
predict = torch.max(output, 1)[1].cpu().numpy() + 1

# Get color pallete for visualization
mask = encoding.utils.get_mask_pallete(predict, 'pascal_voc')
mask.save('output.png')
