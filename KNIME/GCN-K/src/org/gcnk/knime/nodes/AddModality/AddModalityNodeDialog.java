package org.gcnk.knime.nodes.AddModality;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;

/**
 * <code>NodeDialog</code> for the "AddModality" Node.
 *  * nAdd the new modality data to the GCN Dataset with the output of the GCNDatasetBuilder node and the output from the AdditionalModalityPreprocessor node as input
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class AddModalityNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring AddModality node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected AddModalityNodeDialog() {
        super();
        
    }
}

