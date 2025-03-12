import numpy as np
 
def autosave_on(self):
    if self.SCheckBox.isChecked():
        self.autosave = True
    else:
        self.autosave = False

def save_state(self):
    """Save the current state of mask_stack for undo."""
    if len(self.undo_stack) >= self.max_undo_steps:
        self.undo_stack.pop(0)
    self.undo_stack.append(np.copy(self.mask_stack))

def undo_action(self):
    """Undo the last action."""
    if self.undo_stack:
        # Save the current state for redo
        self.redo_stack.append(np.copy(self.mask_stack))
        if len(self.redo_stack) >= self.max_undo_steps:
            self.redo_stack.pop(0)  # Limit redo stack size

        # Restore the last state from the undo stack
        self.mask_stack = self.undo_stack.pop()        
        self.update_layer()  # Refresh the display

    else:
        print("Nothing to undo.")
        
def redo_action(self):
    """Redo the last undone action."""
    if self.redo_stack:

        # Save the current state for undo
        self.undo_stack.append(np.copy(self.mask_stack))
        if len(self.undo_stack) >= self.max_undo_steps:
            self.undo_stack.pop(0)  # Limit undo stack size
            
        # Restore the last state from the redo stack
        self.mask_stack = self.redo_stack.pop()
        self.update_layer()  # Refresh the display
    

    else:
        print("Nothing to redo.")