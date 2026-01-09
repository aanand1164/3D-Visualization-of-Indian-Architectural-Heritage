## 📝 3D Visualization of Indian Architectural Heritage

An AI-driven project that explores the use of generative models and single-view reconstruction techniques to create interactive 3D visualizations of Indian architectural monuments and temples. The project focuses on overcoming data limitations commonly faced by traditional 3D reconstruction pipelines by leveraging learned priors from deep learning models.


## 🚀 Key Components
- Text-guided image generation for architectural structures
- Single-view image-to-3D reconstruction using learned priors
- Point cloud generation followed by 3D mesh reconstruction
- Interactive web-based 3D visualization


## 🛠️ Tech Stack
- **Python** – model inference and reconstruction pipeline  
- **Stable Diffusion 1.5** – text-to-image generation  
- **Point-E** – point cloud generation and 3D reconstruction  
- **Three.js** – real-time 3D rendering  
- **React** – frontend for interactive visualization  
- **Git** – version control  


## 💻 Setup & Execution

1. Clone this repository:
   ```bash
   git clone https://github.com/aanand1164/3D-Visualization-of-Indian-Architectural-Heritage.git
   
2. Set up a Python virtual environment and install required dependencies.

3. Run the text-to-image generation script to generate architectural images.

4. Use the generated images as input for the image-to-3D reconstruction pipeline.

5. Start the React frontend to visualize the reconstructed 3D models interactively in the browser.

Note: Model weights and generated outputs are excluded from version control.


## 📊 Results
The project demonstrates that generative models combined with single-view reconstruction techniques can produce meaningful 3D representations suitable for conceptual visualization and exploratory analysis. While the reconstructed models are not metrically accurate, they effectively capture the structural essence of architectural forms.


## ⚠️ Limitations
- Reconstructed 3D models are not intended for precise geometric or metric measurements.
- Single-view reconstruction limits structural accuracy and fine-grained details.
- Output quality is highly dependent on the generated image quality and model priors.


## 🔮 Future Work
- Incorporating multi-view consistency to improve geometric accuracy
- Exploring neural rendering techniques for enhanced visual quality
- Enabling user-guided refinement and interaction for improved reconstruction control


## 📄 License
This project is licensed under the MIT License.
