package org.gcnk.knime.nodes.SDFReader;

import javax.swing.JFileChooser;
import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponentFileChooser;
import org.knime.core.node.defaultnodesettings.DialogComponentNumber;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;

/**
 * <code>NodeDialog</code> for the "SDFReader" Node.
 * Read labes from CSV file.
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class SDFReaderNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring SDFReader node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected SDFReaderNodeDialog() {
        super();
        
        createNewGroup("Input SDF File");
        addDialogComponent(new DialogComponentFileChooser(
        		new SettingsModelString(SDFReaderNodeModel.CFGKEY_SDF_FILE, ""),
        		"sdfFile", JFileChooser.OPEN_DIALOG, ".sdf"));

        createNewGroup("Options");
        addDialogComponent(new DialogComponentNumber(	
        		new SettingsModelIntegerBounded(
        				SDFReaderNodeModel.CFGKEY_ATOM_NUM_LIMIT,
        				SDFReaderNodeModel.DEFAULT_ATOM_NUM_LIMIT,
        				1, Integer.MAX_VALUE),
        		"Atom Number Limit:", /*step*/ 1));

        createNewGroup("Working Directory");
        addDialogComponent(new DialogComponentFileChooser(
        		new SettingsModelString(SDFReaderNodeModel.CFGKEY_WORK_DIR, ""),
        		"workDir", JFileChooser.OPEN_DIALOG, true));
        
    }
}

