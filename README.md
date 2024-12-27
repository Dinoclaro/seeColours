# SeeColours

### CS50x 2024 Final Project
### By Dino Claro
#### Video Demo: https://youtu.be/7Hnv1ig3hUc

## Overview

SeeColours is a web-based application designed to assist individuals with color vision deficiencies (CVD) by distinguishing colours more effectively and simulating what a person with CVD sees based on a particular prognosis. Having CVD (Deuteranopia) myself, this project is stems from my interest in understanding it as both a medical condition and a computer science challenge.

### Current Development Status
This project is currently in development mode, focusing primarily on core CVD processing functionality rather than production-ready features. While user authentication and database integration are implemented as taught in CS50, these aspects would need significant enhancement for production deployment.

## Key Features

1. **CVD Image Processing**
   - Simulate how images appear to individuals with differnet CVD prognoses
   - Correct images for people with CVD using the Daltonize algorithm
   - Support for custom and generic CVD profiles

2. **User Management**
   - Personal account creation and authentication
   - Storage of individual CVD test results
   - Profile-based image processing preferences

3. **Real-time Processing**
   - AJAX-powered previews
   - Processing status feedback
   - Allows for multiple prognoses and custom-user manipulation

## Technical Architecture

### Core Files and Their Functions

1. **`app.py`**
   - Serves as the main Flask application entry point
   - Manages user sessions and authentication

2. **`helpers.py`**
   - Contains utility functions for:
     - File path management
     - User authentication checks
   - Implements santise measures for uploads

3. **`image_processing.py`**
   - Houses the core CVD simulation and correction algorithms
   - Implements two major models:
     - MacHado, Oliveria and Fernandes (2009) for Deutan and Protan cases
     - Brettle (1997) for Tritan cases

   CVD simulation and Daltonize algorithm
   1. Convert uploaded image from sRGB to linear RGB for accurate color processing
   2. Transform to LMS colorspace, which is calibrated to human cone responses
   3. Project colors along "confusion lines" specific to each type of CVD
   4. For daltonization, calculate and redistribute lost color information
   5. Convert processed image back to standard RGB format

### Template Structure

1. **Base Templates**
   - `layout.html`: Main template with navigation and Bootstrap integration
   - `index.html`: Dashboard showing user's test results and status
   - `about.html`: Educational content about CVD and usage instructions

2. **User Management Templates**
   - `login.html`: User authentication interface
   - `register.html`: New account creation form
   - `submit_test.html`: Interface for CVD test result submission

3. **Processing Templates**
   - `upload.html`: Image upload interface with real-time feedback and image process selection
   - `base_simulator.html`: Universal template for both simulation and correction modes

### Static Resources
- **stom CSS extending Bootstrap**
- **loads folder:**
  - Uloaded and processed images saved to this folder

## Installation and Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the development server:
```bash
flask run
```

## Usage Guide

1. **Account Creation**
   - Register with a unique username and password
   - Passwords are securely hashed before storage

2. **CVD Profile Setup**
   - Take the EnChroma CVD test
   - Submit results via the "Submit Test Result" interface
   - Results are stored for personalized processing

3. **Image Processing**
   - Upload images (PNG/JPEG, max 2MB)
   - Choose processing mode:
     - **Simulate**: View how images appear to individuals with CVD
     - **Daltonize**: Correct images for better color distinction
   - Apply adjustments using either:
     - Personal CVD profile
     - Generic CVD profiles (Protan/Deutan/Tritan)
   - Download processed images

## Design Choices and Rationale

1. **Technology Stack Selection**
   - The technology stack, being the CS50 web app curriculum, was a prior to the project.

2. **Focus on Core Functionality**
   - Prioritized accurate CVD simulation over scalability
   - Implemented real-time AJAX previews for better user experience
   - Created dual-mode processing (simulation and correction)

3. **User Experience Decisions**
   - Included both custom and generic CVD profiles for accessibility
   - Added immediate visual feedback during processing

4. **Security Implementation**
   - Basic password hashing and session management as per CS50
   - Input sanitization for all user-submitted data
   - Simple file type verification for uploads

## Known Limitations

   - CS50’s Flask  environment which by design is in dev mode
   - Basic authentication system (not production-ready)
   - Local file storage (needs cloud integration)
   - Limited testing for edge cases

## Challenges Encountered

1. **Technical Challenges**
   - Understanding color space transformations for accurate simulation
   - Implementing complex CVD models
   - Managing real-time image processing and feedback

## Future Improvements

### Functional Enhancements
1. **Color Analysis Tools**
   - Hover-based color identification
   - Problematic color palette generation
   - Isolate problematic areas of the image

2. **Interface Improvements**
   - Upload history tracking
   - Processing preset management
   - Enhanced visual feedback systems

### Production Readiness
1. **Infrastructure**
   - Cloud-based image storage implementation
   - Migration to a more robust database system
   - Enhanced security measures
   - Proper user authentication system

2. **Performance Optimization**
   - Image processing optimization
   - Processing time reduction
   - Batch processing capabilities

## References

1. DaltonLens.org (https://daltonlens.org/opensource-cvd-simulation/)
   - Primary resource for CVD simulation understanding
   - Model implementation guidelines
   - Algorithm validation methods

2. Machado, G. M., Oliveira, M. M., & Fernandes, L. A. F. (2009). A Physiologically-based Model for Simulation of Color Vision Deficiency. IEEE Transactions on Visualization and Computer Graphics.

3. Brettel, H., Viénot, F., & Mollon, J. D. (1997). Computerized simulation of color appearance for dichromats. Journal of the Optical Society of America A.

---
