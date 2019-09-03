package org.gcnk.knime.nodes.AtomFeatureExtractor;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;

/**
 * <code>NodeDialog</code> for the "AtomFeatureExtractor" Node.
 * Read SDF file and generate atom features.
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class AtomFeatureExtractorNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring AtomFeatureExtractor node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected AtomFeatureExtractorNodeDialog() {
        super();
        
    }
}

