package org.gcnk.knime.nodes.CSVLabelExtractor;

import javax.swing.JFileChooser;
import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponentFileChooser;
import org.knime.core.node.defaultnodesettings.SettingsModelString;

/**
 * <code>NodeDialog</code> for the "CSVLabelExtractor" Node.
 * Read CSV file and extract labels.
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class CSVLabelExtractorNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring CSVLabelExtractor node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected CSVLabelExtractorNodeDialog() {
        super();
        
        createNewGroup("Input CSV File");
        addDialogComponent(new DialogComponentFileChooser(
        		new SettingsModelString(CSVLabelExtractorNodeModel.CFGKEY_CSV_FILE, ""),
        		"csvFile", JFileChooser.OPEN_DIALOG, ".csv"));

    }
}

