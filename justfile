publish-entity:
    cd faer-entity && cargo publish --package faer-entity

publish-libs:
    cd faer-libs   && cargo publish --package faer-core
    cd faer-libs   && cargo publish --package faer-lu
    cd faer-libs   && cargo publish --package faer-qr
    cd faer-libs   && cargo publish --package faer-cholesky
    cd faer-libs   && cargo publish --package faer-svd
    cd faer-libs   && cargo publish --package faer-evd
    cd faer-libs   && cargo publish --package faer
