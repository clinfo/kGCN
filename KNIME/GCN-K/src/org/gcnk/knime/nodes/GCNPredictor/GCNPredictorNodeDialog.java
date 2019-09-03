package org.gcnk.knime.nodes.GCNPredictor;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponentString;
import org.knime.core.node.defaultnodesettings.DialogComponentBoolean;
import org.knime.core.node.defaultnodesettings.DialogComponentNumber;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;

/**
 * <code>NodeDialog</code> for the "GCNPredictor" Node.
 * Predict test dataset with model.
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class GCNPredictorNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring GCNPredictor node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected GCNPredictorNodeDialog() {
        super();
        
        addDialogComponent(new DialogComponentString(
                new SettingsModelString(
                	GCNPredictorNodeModel.CFGKEY_MODEL_PY,
                	GCNPredictorNodeModel.DEFAULT_MODEL_PY),
                	"model.py"));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                	GCNPredictorNodeModel.CFGKEY_WITH_FEATURE,
                	GCNPredictorNodeModel.DEFAULT_WITH_FEATURE),
                    "With Feature"));
        
        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                	GCNPredictorNodeModel.CFGKEY_WITH_NODE_EMBEDDING,
                	GCNPredictorNodeModel.DEFAULT_WITH_NODE_EMBEDDING),
                    "With Node Embedding"));

        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                	GCNPredictorNodeModel.CFGKEY_EMBEDDING_DIM,
                	GCNPredictorNodeModel.DEFAULT_EMBEDDING_DIM,
                    1, Integer.MAX_VALUE),
                	"Embedding Dim", /*step*/ 1, /*componentwidth*/ 5));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                	GCNPredictorNodeModel.CFGKEY_NORMALIZE_ADJ_FLAG,
                	GCNPredictorNodeModel.DEFAULT_NORMALIZE_ADJ_FLAG),
                    "Normalize Adj Flag"));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                	GCNPredictorNodeModel.CFGKEY_SPLIT_ADJ_FLAG,
                	GCNPredictorNodeModel.DEFAULT_SPLIT_ADJ_FLAG),
                    "Split Adj Flag"));

        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                	GCNPredictorNodeModel.CFGKEY_ORDER,
                	GCNPredictorNodeModel.DEFAULT_ORDER,
                    1, Integer.MAX_VALUE),
                    "Order", /*step*/ 1, /*componentwidth*/ 5));

    }
}

