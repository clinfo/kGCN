package org.gcnk.knime.nodes.SDFReader;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "SDFReader" Node.
 * Read labes from CSV file.
 *
 * @author org.gcnk
 */
public class SDFReaderNodeFactory 
        extends NodeFactory<SDFReaderNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public SDFReaderNodeModel createNodeModel() {
        return new SDFReaderNodeModel();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getNrNodeViews() {
        return 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NodeView<SDFReaderNodeModel> createNodeView(final int viewIndex,
            final SDFReaderNodeModel nodeModel) {
        return new SDFReaderNodeView(nodeModel);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean hasDialog() {
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NodeDialogPane createNodeDialogPane() {
        return new SDFReaderNodeDialog();
    }

}

