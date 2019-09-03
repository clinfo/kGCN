package org.gcnk.knime.nodes.GCNDatasetBuilder;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;


/**
 * <code>NodeDialog</code> for the "GCNDatasetBuilder" Node.
 * Read labels from CSV file and Create GCN Dataset together with graph structure and atom features.
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class GCNDatasetBuilderNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring GCNDatasetBuilder node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected GCNDatasetBuilderNodeDialog() {
        super();
        

    }
}

