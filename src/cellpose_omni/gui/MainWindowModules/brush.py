def update_brush_slider_color(self):
    """Update brush slider text color based on pen active state."""
    color = 'red' if self.PencilCheckBox.isChecked() else "gray"
    # print('color',self.accent)
    self.brush_slider.setStyleSheet(f"color: {color};")


def brush_size_change(self):
    """Update the brush size based on slider value."""
    if self.loaded:
        value = self.brush_slider.value()
        
        # make sure this is odd
        odd_value = value | 1  # Ensure the value is odd by setting the least significant bit
        if odd_value != value:  # Only update if the value changes
            self.brush_slider.setValue(odd_value)
            value = odd_value
        
        self.ops_plot = {'brush_size': value}
        self.brush_size = value
        
        self.layer._generateKernel(self.brush_size)
        self.compute_kernel_path(self.layer._kernel)
        self.update_highlight()
        