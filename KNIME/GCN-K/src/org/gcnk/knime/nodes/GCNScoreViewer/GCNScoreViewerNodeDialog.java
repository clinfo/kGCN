package org.gcnk.knime.nodes.GCNScoreViewer;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponentBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;

/**
 * <code>NodeDialog</code> for the "GCNScoreViewer" Node.
 * Show scores from output of GCNScore
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class GCNScoreViewerNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring GCNScoreViewer node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected GCNScoreViewerNodeDialog() {
        super();
        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                		GCNScoreViewerNodeModel.CFGKEY_PLOT_MULTITASK,
                		GCNScoreViewerNodeModel.DEFAULT_PLOT_MULTITASK),
                    "PlotMultitask"));
                    
    }
}

