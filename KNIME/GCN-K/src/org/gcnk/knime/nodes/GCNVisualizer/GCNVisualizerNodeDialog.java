package org.gcnk.knime.nodes.GCNVisualizer;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponentNumber;
import org.knime.core.node.defaultnodesettings.DialogComponentString;
import org.knime.core.node.defaultnodesettings.DialogComponentBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;

/**
 * <code>NodeDialog</code> for the "GCNVisualizer" Node.
 * Generates data for displaying the contribution of each compound of the testset to the predicted value
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class GCNVisualizerNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring GCNVisualizer node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected GCNVisualizerNodeDialog() {
        super();
        
        addDialogComponent(new DialogComponentString(
                new SettingsModelString(
                	GCNVisualizerNodeModel.CFGKEY_MODEL_PY,
                	GCNVisualizerNodeModel.DEFAULT_MODEL_PY),
                    "model.py"));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                	GCNVisualizerNodeModel.CFGKEY_WITH_FEATURE,
                	GCNVisualizerNodeModel.DEFAULT_WITH_FEATURE),
                    "With Feature"));
        
        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                	GCNVisualizerNodeModel.CFGKEY_WITH_NODE_EMBEDDING,
                	GCNVisualizerNodeModel.DEFAULT_WITH_NODE_EMBEDDING),
                    "With Node Embedding"));

        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                	GCNVisualizerNodeModel.CFGKEY_EMBEDDING_DIM,
                    GCNVisualizerNodeModel.DEFAULT_EMBEDDING_DIM,
                    1, Integer.MAX_VALUE),
                	"Embedding Dim", /*step*/ 1, /*componentwidth*/ 5));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                	GCNVisualizerNodeModel.CFGKEY_NORMALIZE_ADJ_FLAG,
                    GCNVisualizerNodeModel.DEFAULT_NORMALIZE_ADJ_FLAG),
                    "Normalize Adj Flag"));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                	GCNVisualizerNodeModel.CFGKEY_SPLIT_ADJ_FLAG,
                    GCNVisualizerNodeModel.DEFAULT_SPLIT_ADJ_FLAG),
                    "Split Adj Flag"));

        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                	GCNVisualizerNodeModel.CFGKEY_ORDER,
                    GCNVisualizerNodeModel.DEFAULT_ORDER,
                    1, Integer.MAX_VALUE),
                    "Order", /*step*/ 1, /*componentwidth*/ 5));
        
    }
}

