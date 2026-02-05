# 📊 Financial Analysis AI Dashboard

A world-class FP&A (Financial Planning & Analysis) AI Agent built with Streamlit that analyzes financial reports, generates insights, and provides forecasting capabilities.

![FP&A AI Dashboard](https://img.shields.io/badge/FP&A-AI%20Dashboard-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)

## ✨ Features

### 🎯 Core Capabilities
- **PDF Financial Report Analysis**: Extract and analyze financial data from PDF reports
- **Multi-Industry Support**: Insurance, Banking, Technology, Airport, and General companies
- **AI-Powered Insights**: Automated financial commentary and ratio analysis
- **Interactive Visualizations**: Waterfall charts, gauges, segment maps, and forecasting graphs
- **IFRS 17 Compliance**: Specialized analysis for insurance companies under IFRS 17

### 📊 Dashboard Components
1. **Overview & KPIs**: Key performance indicators and financial ratios
2. **IFRS 17 Details**: Contractual Service Margin (CSM) analysis and movements
3. **Segment Performance**: Geographic and business segment analysis
4. **Forecasting**: Prophet-based 3-year forecasts
5. **AI Insights**: Template-based financial analysis and recommendations

### 🎨 Design Features
- Light/Dark mode toggle
- Responsive layout
- Professional financial styling
- Interactive charts with Plotly

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

### Requirements
```txt
streamlit==1.32.0
pandas==2.1.4
pdfplumber==0.10.3
plotly==5.18.0
prophet==1.1.5
numpy==1.26.3
python-dateutil==2.8.2
typing-extensions==4.9.0
holidays==0.36
convertdate==2.4.0
lunarcalendar==0.0.9
```

## 📁 Project Structure

```
financial-analysis-dashboard/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore               # Git ignore file
```

## 🖥️ Usage Guide

### 1. Starting the App
```bash
streamlit run app.py
```
The app will open in your default browser at `http://localhost:8501`

### 2. Using the Dashboard

#### Sidebar Controls:
- **Theme Selection**: Switch between Light and Dark modes
- **Company Type**: Select industry (Insurance, Banking, Technology, Airport, Other)
- **PDF Upload**: Upload financial reports for analysis
- **Analysis Modules**: Toggle different dashboard sections
- **Sample Data**: Load pre-built sample datasets

#### Main Dashboard Sections:
1. **Upload PDF**: Upload financial statements for automated extraction
2. **View KPIs**: See key metrics and ratios with interactive gauges
3. **Analyze Segments**: Visualize business unit performance
4. **Generate Forecasts**: Create 3-year forecasts using Prophet
5. **Read AI Insights**: Get automated financial analysis

### 3. Sample Data
The app includes sample data for:
- **Insurance**: Sagicor Financial Company Ltd. (2024)
- **Banking**: CIBC (2024)
- **Technology**: Apple Inc. (2024)
- **Airport**: Grantley Adams International Airport (2024)

## 🔧 Technical Details

### Data Extraction
- Uses `pdfplumber` for PDF text extraction
- Regular expression patterns for financial data capture
- Automatic currency and scale detection
- Support for Barbados Dollars (BBD) and USD

### AI Analysis Layer
- Template-based insights generation
- Historical trend analysis
- Ratio benchmarking
- Industry-specific commentary

### Forecasting
- Facebook Prophet for time series forecasting
- 3-year projections
- Confidence intervals
- Multiple metric support

### Visualization
- Plotly for interactive charts
- Waterfall diagrams for IFRS 17 movements
- Geographic segment mapping
- Gauge charts for ratios

## 🎯 Supported Financial Metrics

### Common Metrics
- Net Income, Revenue, Assets, Equity
- ROE, ROA, Debt-to-Equity, Current Ratio

### Insurance-Specific
- CSM (Contractual Service Margin)
- Risk Adjustment
- Insurance Service Result
- LICAT Ratio

### Banking-Specific
- Net Interest Margin
- Efficiency Ratio
- CET1 Ratio

### Airport-Specific
- Passenger Traffic
- Aircraft Movements
- Cargo/Mail Volumes
- Revenue by Source

## 📈 Future Enhancements

### Planned Features
- [ ] Natural Language Querying ("Ask the AI")
- [ ] Scenario Analysis (Interest rate changes, traffic volume)
- [ ] Automated Variance Analysis vs. Budget
- [ ] Peer Benchmarking Database
- [ ] Advanced NLP for better PDF extraction
- [ ] API Integration for real-time data
- [ ] Custom Report Generation (PDF/Excel)

### Technical Roadmap
- [ ] Database integration for historical data
- [ ] User authentication and multi-tenant support
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] API endpoints for programmatic access
- [ ] Mobile-responsive design improvements

## 🛠️ Development

### Running in Development Mode
```bash
streamlit run app.py --server.runOnSave true
```

### Testing PDF Extraction
The app includes basic regex patterns for financial data extraction. For custom reports, you may need to:
1. Add new regex patterns in `extract_financial_data()` function
2. Test with sample PDFs from your industry
3. Adjust company type configurations

### Adding New Company Types
1. Add to `company_type_options` in sidebar
2. Create sample data function (like `get_apple_sample_data()`)
3. Add industry-specific ratio configurations
4. Update insight generation templates

## ⚠️ Known Limitations

### Current Limitations
- Basic PDF extraction (may not capture all data points)
- Template-based insights (not true AI generation)
- Limited historical data in samples
- No database persistence
- Manual pattern adjustment needed for new report formats

### Dependencies to Note
- Prophet can be tricky to install on some systems
- PDF extraction quality depends on PDF structure
- Some financial ratios use industry-specific calculations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Contribution
1. Improved PDF extraction patterns
2. Additional industry templates
3. Enhanced visualization components
4. Better error handling
5. Performance optimizations

## 📄 License

This project is created for educational and demonstration purposes.

## 👥 Authors

**Matthew Blackman** with assistance from **DeepSeek AI**

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Facebook Prophet for time series forecasting
- Plotly for interactive visualizations
- The open-source community for various libraries used

## 📞 Support

For issues or questions:
1. Check the [Known Limitations](#-known-limitations) section
2. Ensure all dependencies are properly installed
3. Try with sample data first to verify functionality

---

*Note: This is a demonstration application. Financial analysis should be verified by qualified professionals.*
