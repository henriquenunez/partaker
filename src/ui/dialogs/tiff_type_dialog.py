from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QLabel,
    QPushButton, QButtonGroup, QGroupBox, QTextEdit
)
from PySide6.QtCore import Qt

class TIFFTypeDialog(QDialog):
    def __init__(self, tiff_files, parent=None):
        super().__init__(parent)
        self.tiff_files = tiff_files
        self.selected_type = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Select TIFF File Type")
        self.setFixedSize(600, 500)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"Found {len(self.tiff_files)} TIFF file(s). Please select the file organization type:")
        title.setWordWrap(True)
        layout.addWidget(title)
        
        # File type selection
        type_group = QGroupBox("File Organization Type")
        type_layout = QVBoxLayout()
        
        self.button_group = QButtonGroup()
        
        # Single sequence option
        self.single_radio = QRadioButton("Single TIFF Sequence")
        self.single_radio.setChecked(True)  # Default selection
        type_layout.addWidget(self.single_radio)
        
        single_desc = QLabel("Multiple separate TIFF files, each containing one frame.\nNaming convention: image_001.tif, image_002.tif, image_003.tif...")
        single_desc.setStyleSheet("color: gray; margin-left: 20px; margin-bottom: 10px;")
        single_desc.setWordWrap(True)
        type_layout.addWidget(single_desc)
        
        # Multi-frame option
        self.multiframe_radio = QRadioButton("Multi-frame TIFF")
        type_layout.addWidget(self.multiframe_radio)
        
        multi_desc = QLabel("Single TIFF file containing multiple frames/time points.\nExample: timelapse.tif (contains all time points)")
        multi_desc.setStyleSheet("color: gray; margin-left: 20px; margin-bottom: 10px;")
        multi_desc.setWordWrap(True)
        type_layout.addWidget(multi_desc)
        
        # Series option
        self.series_radio = QRadioButton("TIFF Series")
        type_layout.addWidget(self.series_radio)
        
        series_desc = QLabel("Multiple TIFF files with position/time/channel information in filenames.\nNaming convention: pos_001_t_000_ch_0.tif, pos_001_t_001_ch_0.tif...")
        series_desc.setStyleSheet("color: gray; margin-left: 20px; margin-bottom: 10px;")
        series_desc.setWordWrap(True)
        type_layout.addWidget(series_desc)
        
        # Add radio buttons to button group
        self.button_group.addButton(self.single_radio, 0)
        self.button_group.addButton(self.multiframe_radio, 1)
        self.button_group.addButton(self.series_radio, 2)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Selected files preview
        preview_group = QGroupBox("Selected Files Preview")
        preview_layout = QVBoxLayout()
        
        self.file_preview = QTextEdit()
        self.file_preview.setMaximumHeight(120)
        self.file_preview.setReadOnly(True)
        
        # Show first 10 files
        preview_text = "\n".join([f.split('/')[-1] for f in self.tiff_files[:10]])
        if len(self.tiff_files) > 10:
            preview_text += f"\n... and {len(self.tiff_files) - 10} more files"
        self.file_preview.setText(preview_text)
        
        preview_layout.addWidget(self.file_preview)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def get_selected_type(self):
        """Return the selected TIFF type"""
        if self.single_radio.isChecked():
            return "single_sequence"
        elif self.multiframe_radio.isChecked():
            return "multiframe"
        elif self.series_radio.isChecked():
            return "series"
        return None
        
    def accept(self):
        self.selected_type = self.get_selected_type()
        
        # Validate filenames for selected type
        if not self.validate_filenames():
            return  # Don't close dialog if validation fails
            
        super().accept()
    
    def validate_filenames(self):
        """Validate that filenames match the expected pattern for selected type"""
        from PySide6.QtWidgets import QMessageBox
        import re
        
        selected_type = self.get_selected_type()
        
        if selected_type == "series":
            # Check for pos_XXX_t_XXX_ch_X.tif pattern
            pattern = r'pos_(\d+)_t_(\d+)_ch_(\d+)\.tiff?$'
            invalid_files = []
            
            for file_path in self.tiff_files:
                filename = file_path.split('/')[-1]
                if not re.search(pattern, filename):
                    invalid_files.append(filename)
            
            if invalid_files:
                QMessageBox.warning(
                    self, "Invalid Filenames", 
                    f"The following files don't match the expected pattern for TIFF Series:\n\n"
                    f"Expected: pos_001_t_000_ch_0.tif\n\n"
                    f"Invalid files:\n" + "\n".join(invalid_files[:5]) + 
                    (f"\n... and {len(invalid_files) - 5} more" if len(invalid_files) > 5 else "")
                )
                return False
        
        elif selected_type == "single_sequence":
            # Check for simple numbering pattern (flexible)
            # Accept: image_001.tif, frame_0001.tif, etc.
            pattern = r'.*\d+\.tiff?$'
            invalid_files = []
            
            for file_path in self.tiff_files:
                filename = file_path.split('/')[-1]
                if not re.search(pattern, filename):
                    invalid_files.append(filename)
            
            if invalid_files:
                QMessageBox.warning(
                    self, "Invalid Filenames",
                    f"The following files don't contain numbers for sequence ordering:\n\n"
                    f"Expected: image_001.tif, frame_0001.tif, etc.\n\n"
                    f"Invalid files:\n" + "\n".join(invalid_files[:5]) +
                    (f"\n... and {len(invalid_files) - 5} more" if len(invalid_files) > 5 else "")
                )
                return False
        
        # Multi-frame doesn't need filename validation (single file)
        return True