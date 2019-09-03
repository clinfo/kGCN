package org.gcnk.knime.nodes.AtomFeatureExtractor;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "AtomFeatureExtractor" Node.
 * Read SDF file and generate atom features.
 *
 * @author org.gcnk
 */
public class AtomFeatureExtractorNodeFactory 
        extends NodeFactory<AtomFeatureExtractorNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public AtomFeatureExtractorNodeModel createNodeModel() {
        return new AtomFeatureExtractorNodeModel();
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
    public NodeView<AtomFeatureExtractorNodeModel> createNodeView(final int viewIndex,
            final AtomFeatureExtractorNodeModel nodeModel) {
        return new AtomFeatureExtractorNodeView(nodeModel);
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
        return new AtomFeatureExtractorNodeDialog();
    }

}

