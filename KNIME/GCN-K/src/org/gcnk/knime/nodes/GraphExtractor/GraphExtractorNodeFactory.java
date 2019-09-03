package org.gcnk.knime.nodes.GraphExtractor;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "GraphExtractor" Node.
 * Read SDF file and extract graph structures
 *
 * @author org.gcnk
 */
public class GraphExtractorNodeFactory 
        extends NodeFactory<GraphExtractorNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public GraphExtractorNodeModel createNodeModel() {
        return new GraphExtractorNodeModel();
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
    public NodeView<GraphExtractorNodeModel> createNodeView(final int viewIndex,
            final GraphExtractorNodeModel nodeModel) {
        return new GraphExtractorNodeView(nodeModel);
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
        return new GraphExtractorNodeDialog();
    }

}

