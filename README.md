# FCS Entreport

Enterprise Reporting using Streamlit

## Overview

FCS Entreport is an enterprise-level reporting application built with [Streamlit](https://streamlit.io/). It enables users to create, view, and manage various business reports through a modern, interactive web interface.

The main entry point for the application is [`mainui.py`](mainui.py).

## Features

- Interactive web-based reporting dashboard
- User-friendly interface powered by Streamlit
- Customizable and extendable report generation
- Secure and scalable for enterprise use-cases

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/szanislaw/fcs-entreport.git
   cd fcs-entreport
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the main Streamlit application:

```bash
streamlit run mainui.py
```

This will start the FCS Entreport dashboard in your default web browser.

## Project Structure

```
fcs-entreport/
│
├── mainui.py           # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── ...                 # Additional modules and resources
```

## Usage

- Access the dashboard via your browser after running the `streamlit run mainui.py` command.
- Follow on-screen instructions to use the reporting features.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

---

*Enterprise Reporting made simple with Streamlit.*