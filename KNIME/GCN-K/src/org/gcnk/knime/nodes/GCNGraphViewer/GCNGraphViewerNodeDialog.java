package org.gcnk.knime.nodes.GCNGraphViewer;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;

/**
 * <code>NodeDialog</code> for the "GCNGraphViewer" Node.
 * Display graphically the contribution of each atom of each test compound in the test set to the predicted value on the compound from the output of the GCNVisualizer
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class GCNGraphViewerNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring GCNGraphViewer node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected GCNGraphViewerNodeDialog() {
        super();
        
                    
    }
}

