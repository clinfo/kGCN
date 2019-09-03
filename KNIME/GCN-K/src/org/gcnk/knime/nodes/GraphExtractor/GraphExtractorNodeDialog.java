package org.gcnk.knime.nodes.GraphExtractor;

import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;

/**
 * <code>NodeDialog</code> for the "GraphExtractor" Node.
 * Read SDF file and extract graph structures
 *
 * This node dialog derives from {@link DefaultNodeSettingsPane} which allows
 * creation of a simple dialog with standard components. If you need a more 
 * complex dialog please derive directly from 
 * {@link org.knime.core.node.NodeDialogPane}.
 * 
 * @author org.gcnk
 */
public class GraphExtractorNodeDialog extends DefaultNodeSettingsPane {

    /**
     * New pane for configuring GraphExtractor node dialog.
     * This is just a suggestion to demonstrate possible default dialog
     * components.
     */
    protected GraphExtractorNodeDialog() {
        super();
        
    }
}

