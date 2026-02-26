"""
PDF report generation utility for inference results.
"""
import os
import requests
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from io import BytesIO
import logging

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("reportlab not available. PDF generation will not work.")

logger = logging.getLogger(__name__)


def download_image(url: str, timeout: int = 30) -> Image.Image:
    """
    Download an image from URL and return as PIL Image.
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
    
    Returns:
        PIL Image object
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise


def interpret_metrics(metrics: Dict[str, float]) -> str:
    """
    Generate interpretation summary based on metrics.
    
    Args:
        metrics: Dictionary with wt_volume_cc, tc_volume_cc, et_volume_cc
    
    Returns:
        Interpretation text
    """
    wt = metrics.get("wt_volume_cc", 0)
    tc = metrics.get("tc_volume_cc", 0)
    et = metrics.get("et_volume_cc", 0)
    
    interpretations = []
    
    # Whole Tumor assessment
    if wt == 0:
        interpretations.append("No tumor tissue detected in the segmentation.")
    elif wt < 1:
        interpretations.append(f"Very small whole tumor volume ({wt:.2f} cc).")
    elif wt < 10:
        interpretations.append(f"Small whole tumor volume ({wt:.2f} cc).")
    elif wt < 50:
        interpretations.append(f"Moderate whole tumor volume ({wt:.2f} cc).")
    else:
        interpretations.append(f"Large whole tumor volume ({wt:.2f} cc).")
    
    # Tumor Core assessment
    if tc > 0:
        tc_ratio = (tc / wt * 100) if wt > 0 else 0
        interpretations.append(f"Tumor core volume: {tc:.2f} cc ({tc_ratio:.1f}% of whole tumor).")
    
    # Enhancing Tumor assessment
    if et > 0:
        et_ratio = (et / wt * 100) if wt > 0 else 0
        interpretations.append(f"Enhancing tumor volume: {et:.2f} cc ({et_ratio:.1f}% of whole tumor).")
        if et_ratio > 20:
            interpretations.append("Significant enhancing component present.")
    
    # Overall summary
    if wt == 0:
        summary = "No tumor detected."
    elif et > 0:
        summary = "Tumor with enhancing component identified."
    elif tc > 0:
        summary = "Tumor core identified without significant enhancement."
    else:
        summary = "Minimal tumor tissue detected."
    
    return f"{summary}\n\n" + "\n".join(interpretations)


def generate_report_pdf(
    job_id: str,
    outputs: Dict[str, Any],
    metrics: Dict[str, float],
    output_dir: str,
    base_url: str = "http://localhost:8000"
) -> str:
    """
    Generate a PDF report for inference results.
    
    Args:
        job_id: Job identifier
        outputs: Dictionary with output URLs (montage_axial, montage_coronal, montage_sagittal, etc.)
        metrics: Dictionary with wt_volume_cc, tc_volume_cc, et_volume_cc
        output_dir: Directory to save the PDF
        base_url: Base URL for generating download links
    
    Returns:
        URL to the generated PDF file
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # PDF file path
    pdf_filename = f"{job_id}_report.pdf"
    pdf_path = output_path / pdf_filename
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for PDF elements
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 14
    
    # Title
    story.append(Paragraph("Brain Tumor Segmentation Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Job ID
    story.append(Paragraph(f"<b>Job ID:</b> {job_id}", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Metrics Section
    story.append(Paragraph("Volume Metrics", heading_style))
    
    metrics_data = [
        ['Metric', 'Volume (cc)'],
        ['Whole Tumor (WT)', f"{metrics.get('wt_volume_cc', 0):.2f}"],
        ['Tumor Core (TC)', f"{metrics.get('tc_volume_cc', 0):.2f}"],
        ['Enhancing Tumor (ET)', f"{metrics.get('et_volume_cc', 0):.2f}"],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Interpretation
    story.append(Paragraph("Interpretation", heading_style))
    interpretation_text = interpret_metrics(metrics)
    story.append(Paragraph(interpretation_text.replace('\n', '<br/>'), normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Montage Images Section
    story.append(Paragraph("Segmentation Montages", heading_style))
    
    montage_views = [
        ("Axial View", outputs.get("montage_axial")),
        ("Coronal View", outputs.get("montage_coronal")),
        ("Sagittal View", outputs.get("montage_sagittal")),
    ]
    
    for view_name, montage_url in montage_views:
        if not montage_url:
            continue
        
        try:
            # Download and convert image
            img = download_image(montage_url)
            
            # Resize to fit page (max width 6.5 inches, maintain aspect ratio)
            max_width = 6.5 * inch
            img_width, img_height = img.size
            aspect_ratio = img_height / img_width
            
            if img_width > max_width:
                display_width = max_width
                display_height = max_width * aspect_ratio
            else:
                display_width = img_width * (72 / 96)  # Convert pixels to points (assuming 96 DPI)
                display_height = img_height * (72 / 96)
            
            # Save image to temporary file for reportlab
            temp_img_path = output_path / f"temp_{view_name.lower().replace(' ', '_')}.png"
            img.save(temp_img_path)
            
            # Add to PDF
            story.append(Paragraph(f"<b>{view_name}</b>", normal_style))
            story.append(Spacer(1, 0.1*inch))
            
            rl_img = RLImage(str(temp_img_path), width=display_width, height=display_height)
            story.append(rl_img)
            story.append(Spacer(1, 0.2*inch))
            
            # Clean up temp file
            if temp_img_path.exists():
                temp_img_path.unlink()
                
        except Exception as e:
            logger.error(f"Failed to add {view_name} to PDF: {e}")
            story.append(Paragraph(f"<i>Error loading {view_name}</i>", normal_style))
            story.append(Spacer(1, 0.2*inch))
    
    # Footer
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        f"<i>Report generated by NeuroVision Inference Server</i>",
        ParagraphStyle('Footer', parent=normal_style, fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    ))
    
    # Build PDF
    doc.build(story)
    
    logger.info(f"PDF report generated: {pdf_path}")
    
    # Return URL
    pdf_url = f"{base_url}/api/download/{job_id}/{pdf_filename}"
    return pdf_url

