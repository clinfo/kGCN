package org.gcnk.knime.nodes.AdditionalModalityPreprocessor;

import javax.swing.JFileChooser;
import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponentFileChooser;
import org.knime.core.node.defaultnodesettings.DialogComponentButtonGroup;
import org.knime.core.node.defaultnodesettings.SettingsModelString;

/**
 * <code>NodeDialog</code> for the "AdditionalModalityPreprocessor" Node.
 * Read new modality in CSV file format and generates output for AddModality node
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class AdditionalModalityPreprocessorNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring AdditionalModalityPreprocessor node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected AdditionalModalityPreprocessorNodeDialog() {
        super();
                
        addDialogComponent(new DialogComponentButtonGroup(
        		new SettingsModelString(AdditionalModalityPreprocessorNodeModel.CFGKEY_MODALITY, ""),
        		false, "Modality", "profeat", "sequence"));

        createNewGroup("Input CSV File");
        addDialogComponent(new DialogComponentFileChooser(
        		new SettingsModelString(AdditionalModalityPreprocessorNodeModel.CFGKEY_CSV_FILE, ""),
        		"csvFile", JFileChooser.OPEN_DIALOG, ".csv"));
                    
    }
}

