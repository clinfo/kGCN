package org.gcnk.knime.nodes.GCNLearner;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponentNumber;
import org.knime.core.node.defaultnodesettings.DialogComponentNumberEdit;
import org.knime.core.node.defaultnodesettings.DialogComponentString;
import org.knime.core.node.defaultnodesettings.DialogComponentBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;
import org.knime.core.node.defaultnodesettings.SettingsModelDoubleBounded;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;

/**
 * <code>NodeDialog</code> for the "GCNLearner" Node.
 * Generate prediction model with training dataset.
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class GCNLearnerNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring GCNLearner node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected GCNLearnerNodeDialog() {
        super();
        
        addDialogComponent(new DialogComponentString(
                new SettingsModelString(
                    GCNLearnerNodeModel.CFGKEY_MODEL_PY,
                    GCNLearnerNodeModel.DEFAULT_MODEL_PY),
                    "model.py"));

        addDialogComponent(new DialogComponentNumber(
                new SettingsModelDoubleBounded(
                    GCNLearnerNodeModel.CFGKEY_VALIDATION_DATA_RATE,
                    GCNLearnerNodeModel.DEFAULT_VALIDATION_DATA_RATE,
                    0.0, 1.0),
                    "Validation Data Rate", /*step*/ 0.1, /*componentwidth*/ 5));

        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                    GCNLearnerNodeModel.CFGKEY_EPOCH,
                    GCNLearnerNodeModel.DEFAULT_EPOCH,
                    1, Integer.MAX_VALUE),
                    "Epoch", /*step*/ 1, /*componentwidth*/ 5));

        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                    GCNLearnerNodeModel.CFGKEY_BATCH_SIZE,
                    GCNLearnerNodeModel.DEFAULT_BATCH_SIZE,
                    1, Integer.MAX_VALUE),
                    "Batch Size", /*step*/ 1, /*componentwidth*/ 5));
        
        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                    GCNLearnerNodeModel.CFGKEY_PATIENCE,
                    GCNLearnerNodeModel.DEFAULT_PATIENCE,
                    0, Integer.MAX_VALUE),
                    "Patience", /*step*/ 1, /*componentwidth*/ 5));

        addDialogComponent(new DialogComponentNumberEdit(
                new SettingsModelDoubleBounded(
                    GCNLearnerNodeModel.CFGKEY_LEARNING_RATE,
                    GCNLearnerNodeModel.DEFAULT_LEARNING_RATE,
                    0.0, 1.0),
                    "Learning Rate", /*componentwidth*/ 5));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                    GCNLearnerNodeModel.CFGKEY_SHUFFLE_DATA,
                    GCNLearnerNodeModel.DEFAULT_SHUFFLE_DATA),
                    "Shuffle Data"));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                    GCNLearnerNodeModel.CFGKEY_WITH_FEATURE,
                    GCNLearnerNodeModel.DEFAULT_WITH_FEATURE),
                    "With Feature"));
        
        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                    GCNLearnerNodeModel.CFGKEY_WITH_NODE_EMBEDDING,
                    GCNLearnerNodeModel.DEFAULT_WITH_NODE_EMBEDDING),
                    "With Node Embedding"));

        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                    GCNLearnerNodeModel.CFGKEY_EMBEDDING_DIM,
                    GCNLearnerNodeModel.DEFAULT_EMBEDDING_DIM,
                    1, Integer.MAX_VALUE),
                	"Embedding Dim", /*step*/ 1, /*componentwidth*/ 5));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                    GCNLearnerNodeModel.CFGKEY_NORMALIZE_ADJ_FLAG,
                    GCNLearnerNodeModel.DEFAULT_NORMALIZE_ADJ_FLAG),
                    "Normalize Adj Flag"));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                    GCNLearnerNodeModel.CFGKEY_SPLIT_ADJ_FLAG,
                    GCNLearnerNodeModel.DEFAULT_SPLIT_ADJ_FLAG),
                    "Split Adj Flag"));

        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                    GCNLearnerNodeModel.CFGKEY_ORDER,
                    GCNLearnerNodeModel.DEFAULT_ORDER,
                    1, Integer.MAX_VALUE),
                    "Order", /*step*/ 1, /*componentwidth*/ 5));
        
        addDialogComponent(new DialogComponentNumber(
                new SettingsModelIntegerBounded(
                    GCNLearnerNodeModel.CFGKEY_SAVE_INTERVAL,
                    GCNLearnerNodeModel.DEFAULT_SAVE_INTERVAL,
                    1, Integer.MAX_VALUE),
                    "Save Interval", /*step*/ 1, /*componentwidth*/ 5));
        
        
        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                    GCNLearnerNodeModel.CFGKEY_MAKE_PLOT,
                    GCNLearnerNodeModel.DEFAULT_MAKE_PLOT),
                    "Make Plot"));

        addDialogComponent(new DialogComponentBoolean(
                new SettingsModelBoolean(
                    GCNLearnerNodeModel.CFGKEY_PROFILE,
                    GCNLearnerNodeModel.DEFAULT_PROFILE),
                    "Profile"));
    }
}

