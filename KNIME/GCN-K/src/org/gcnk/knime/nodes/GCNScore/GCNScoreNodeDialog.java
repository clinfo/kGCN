package org.gcnk.knime.nodes.GCNScore;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;

/**
 * <code>NodeDialog</code> for the "GCNScore" Node.
 * Calculate scores from output of GCNPredictior
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class GCNScoreNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring GCNScore node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected GCNScoreNodeDialog() {
        super();
        
    }
}

