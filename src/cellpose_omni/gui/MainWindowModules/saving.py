import numpy as np
from .. import logger
def autosave_on(self):
    if self.PencilCheckBox.isChecked():
        self.autosave = True
    else:
        self.autosave = False

def save_state(self):
    """Save the current state of mask_stack, outl_stack, csum, and affinity_graph for undo."""
    if len(self.undo_stack) >= self.max_undo_steps:
        self.undo_stack.pop(0)

    state_dict = {
        'mask_stack': np.copy(self.mask_stack),
        'outl_stack': np.copy(self.outl_stack),
        'csum': np.copy(self.csum),
        'affinity_graph': np.copy(self.affinity_graph) if self.affinity_graph is not None else None,
    }

    self.undo_stack.append(state_dict)

def undo_action(self):
    """Undo the last action, restoring mask_stack, outl_stack, csum, and affinity_graph from undo stack."""
    if self.undo_stack:
        # Save the current state for redo
        redo_dict = {
            'mask_stack': np.copy(self.mask_stack),
            'outl_stack': np.copy(self.outl_stack),
            'csum': np.copy(self.csum),
            'affinity_graph': np.copy(self.affinity_graph) if self.affinity_graph is not None else None,
        }
        self.redo_stack.append(redo_dict)
        if len(self.redo_stack) >= self.max_undo_steps:
            self.redo_stack.pop(0)

        # Restore from the undo stack
        last_state = self.undo_stack.pop()
        self.mask_stack = last_state['mask_stack']
        self.outl_stack = last_state['outl_stack']
        self.csum = last_state['csum']
        self.affinity_graph = last_state['affinity_graph']
        self.update_layer_and_graph()  # Refresh the display

    else:
        print("Nothing to undo.")
        
def redo_action(self):
    """Redo the last undone action, restoring mask_stack, outl_stack, csum, and affinity_graph from redo stack."""
    if self.redo_stack:
        # Save the current state for undo
        undo_dict = {
            'mask_stack': np.copy(self.mask_stack),
            'outl_stack': np.copy(self.outl_stack),
            'csum': np.copy(self.csum),
            'affinity_graph': np.copy(self.affinity_graph) if self.affinity_graph is not None else None,
        }
        self.undo_stack.append(undo_dict)
        if len(self.undo_stack) >= self.max_undo_steps:
            self.undo_stack.pop(0)

        # Restore from the redo stack
        next_state = self.redo_stack.pop()
        self.mask_stack = next_state['mask_stack']
        self.outl_stack = next_state['outl_stack']
        self.csum = next_state['csum']
        self.affinity_graph = next_state['affinity_graph']
        self.update_layer_and_graph()
        
            
    else:
        print("Nothing to redo.")