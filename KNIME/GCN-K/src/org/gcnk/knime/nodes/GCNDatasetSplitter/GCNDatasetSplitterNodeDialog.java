package org.gcnk.knime.nodes.GCNDatasetSplitter;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponentNumber;
import org.knime.core.node.defaultnodesettings.SettingsModelDoubleBounded;

/**
 * <code>NodeDialog</code> for the "GCNDatasetSplitter" Node.
 * Split dataset into two parts.
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class GCNDatasetSplitterNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring GCNDatasetSplitter node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected GCNDatasetSplitterNodeDialog() {
        super();
        
        addDialogComponent(new DialogComponentNumber(
                new SettingsModelDoubleBounded(
                    GCNDatasetSplitterNodeModel.CFGKEY_RATIO,
                    GCNDatasetSplitterNodeModel.DEFAULT_RATIO,
                    0.0, 1.0),
                    "Ratio", /*step*/ 0.1, /*componentwidth*/ 5));

    }
}

