This repository provides the official implementation of SCaRNet, a circular representation-based detection framework for planetary crater detection.
SCaRNet formulates crater detection using a center–radius representation, enabling a more compact and geometrically consistent alternative to conventional bounding-box and segmentation-based methods.
We provide pretrained weights for both server-based experiments and embedded deployment.

🔹 Server (Full-precision model)
Suitable for training and evaluation on GPUs
Download: [Google Drive Link - Server Weights]

🔹 Embedded (Atlas 200I DK A2)
Optimized for deployment on the Atlas 200I DK A2 platform

Includes quantized / deployment-ready models
Download: [[Google Drive Link - Embedded Weights]](https://drive.google.com/file/d/185H3-H68JjKT4pPNqGpsyjYmTGPS6Jqr/view?usp=sharing)

⚠️ Please refer to the deployment scripts for hardware-specific inference settings.


## 🙏 Acknowledgments
Parts of this codebase are adapted from or inspired by the CircleNet framework. We sincerely thank the authors for their open-source contributions.
